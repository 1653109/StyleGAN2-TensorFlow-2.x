import os
import time
import argparse
import numpy as np
import tensorflow as tf

from utils.dataset_loader import DatasetLoader
from utils.utils_stylegan2 import postprocess_images, EasyDict, merge_batch_images, lerp
from stylegan2_generator import StyleGan2Generator
from stylegan2_discriminator import StyleGan2Discriminator

class Trainer(object):
    def __init__(
        self,
        model_base_dir      = './models',
        datasets_dir        = './datasets',
        shuffle_buffer_size = 1000,
        max_to_keep         = 2,
        g_kwargs            = {},
        d_kwargs            = {},
        g_opt               = {},
        d_opt               = {},
        batch_size          = 16,
        n_total_image       = 25000000,
        n_samples           = 4,
        lazy_regularization  = True,
        name                = "stylegan2"):

        self.model_base_dir = model_base_dir
        self.g_kwargs = g_kwargs
        self.d_kwargs = d_kwargs
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.batch_size = batch_size
        self.n_total_image = n_total_image
        self.n_samples = min(self.batch_size, n_samples)
        self.lazy_regularization = lazy_regularization

        self.r1_gamma = 10.0
        self.max_steps = int(np.ceil(self.n_total_image / self.batch_size))
        self.out_res = self.g_kwargs['resolution']
        self.log_template = 'step {}: elapsed: {:.2f}s, d_loss: {:.3f}, g_loss: {:.3f}, r1_reg: {:.3f}, pl_reg: {:.3f}'
        # self.log_template = 'step {}: elapsed: {:.2f}s, d_loss: {:.3f}, g_loss: {:.3f}, gradient_penalty: {:.3f}, pl_reg: {:.3f}'
        self.print_step = 100
        self.save_step = 1000
        self.image_summary_step = 1000
        self.reached_max_steps = False

        # set up optimizer params
        self.g_opt = self.set_optimizer_params(self.g_opt)
        self.d_opt = self.set_optimizer_params(self.d_opt)
        self.pl_mean = tf.Variable(initial_value=0.0, name='pl_mean', trainable=False)
        self.pl_decay = 0.01
        self.pl_weight = 1.0
        self.pl_denorm = 1.0 / np.sqrt(self.out_res * self.out_res)

        # dataset
        self.dataset_loader = DatasetLoader(path_dir=datasets_dir, resolution=self.out_res, batch_size=self.batch_size, shuffle_buffer_size=shuffle_buffer_size)

        # init models
        print('Initializing models ...')
        self.generator = StyleGan2Generator(**self.g_kwargs)
        self.discriminator = StyleGan2Discriminator(**self.d_kwargs)
        
        self.g_optimizer = tf.keras.optimizers.Adam(self.g_opt['learning_rate'], beta_1=self.g_opt['beta1'], beta_2=self.g_opt['beta2'], epsilon=self.g_opt['epsilon'])
        self.d_optimizer = tf.keras.optimizers.Adam(self.d_opt['learning_rate'], beta_1=self.d_opt['beta1'], beta_2=self.d_opt['beta2'], epsilon=self.g_opt['epsilon'])

        self.g_clone = StyleGan2Generator(**self.g_kwargs)

        # set up params for evaluate ...
        test_latent = np.ones((1, self.g_kwargs['z_dim']), dtype=np.float32)
        test_labels = np.ones((1, self.g_kwargs['labels_dim']), dtype=np.float32)
        test_images = np.ones((1, 3, self.out_res, self.out_res), dtype=np.float32)
        __ = self.generator(test_latent, test_labels, training=False)
        _ = self.discriminator(test_images, test_labels, training=False)
        __ = self.g_clone(test_latent, test_labels, training=False)

        # copying Gs
        self.g_clone.set_weights(self.generator.get_weights())

        # setup saving locations
        self.ckpt_dir = os.path.join(self.model_base_dir, name)
        self.ckpt = tf.train.Checkpoint(d_optimizer=self.d_optimizer,
                                        g_optimizer=self.g_optimizer,
                                        discriminator=self.discriminator,
                                        generator=self.generator,
                                        g_clone=self.g_clone)
        
        self.manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=max_to_keep)

        # restore ckpt
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print('Restoring from {}'.format(self.manager.latest_checkpoint))

            # check if any trained
            restored_step = self.g_optimizer.iterations.numpy()
            if restored_step >= self.max_steps:
                print('Already reached max steps {}/{}'.format(restored_step, self.max_steps))
                self.reached_max_steps = True
                return

        else:
            print('Cannot restore from saved checkpoint')

    def set_optimizer_params(self, params):
        if self.lazy_regularization:
            mb_ratio = params['reg_interval'] / (params['reg_interval'] + 1)
            params['learning_rate'] = params['learning_rate'] * mb_ratio
            params['beta1'] = params['beta1'] ** mb_ratio
            params['beta2'] = params['beta2'] ** mb_ratio
        return params

    @tf.function
    def d_train_step(self, z, real_images, labels):
        with tf.GradientTape() as d_tape:
            # fw pass
            fake_images = self.generator(z, labels, training=True)
            real_scores = self.discriminator(real_images, labels, training=True)
            fake_scores = self.discriminator(fake_images, labels, training=True)

            # gan loss
            d_loss = tf.math.softplus(fake_scores)
            d_loss += tf.math.softplus(-real_scores)
            d_loss = tf.reduce_mean(d_loss)
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        return d_loss

    @tf.function
    def d_reg_train_step(self, z, real_images, labels):
        with tf.GradientTape() as d_tape:
            # fw pass
            fake_images = self.generator(z, labels, training=True)
            real_scores = self.discriminator(real_images, labels, training=True)
            fake_scores = self.discriminator(fake_images, labels, training=True)

            # gan loss
            d_loss = tf.math.softplus(fake_scores)
            d_loss += tf.math.softplus(-real_scores)

            # GP
            with tf.GradientTape() as p_tape:
                p_tape.watch(real_images)
                real_loss = tf.reduce_sum(self.discriminator(real_images, labels, training=True))

            real_grads = p_tape.gradient(real_loss, real_images)
            r1_penalty = tf.reduce_sum(tf.math.square(real_grads), axis=[1, 2, 3])
            r1_penalty = tf.expand_dims(r1_penalty, axis=1)
            r1_penalty = r1_penalty * self.d_opt['reg_interval']

            # combine
            d_loss += r1_penalty * (0.5 * self.r1_gamma)
            d_loss = tf.reduce_mean(d_loss)
        
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        return d_loss, tf.reduce_mean(r1_penalty)

    @tf.function
    def d_wgan_gp(self, z, real_images, labels, wgan_lambda=10.0, wgan_epsilon=0.001, wgan_target=1.0):
        with tf.GradientTape() as d_tape:
            # fw pass
            fake_images = self.generator(z, labels, training=True)
            real_scores = self.discriminator(real_images, labels, training=True)
            fake_scores = self.discriminator(fake_images, labels, training=True)

            # gan loss
            d_loss = tf.math.softplus(fake_scores)
            d_loss += tf.math.softplus(-real_scores)

            # epsilon penalty
            # epsilon_penalty = tf.square(real_scores) * wgan_epsilon
            # loss += epsilon_penalty * wgan_epsilon

            # calc gradient penalty
            alpha = tf.random.uniform([z.shape[0], 1, 1, 1], 0., 1.)
            mixed_images = lerp(real_images, fake_images, alpha)
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(mixed_images)
                mixed_scores = self.discriminator(mixed_images, labels, training=True)
            mixed_gradient = gp_tape.gradient(mixed_scores, [mixed_images])[0]
            mixed_slopes = tf.sqrt(tf.reduce_sum(tf.square(mixed_gradient), axis=[1, 2, 3]))
            gradient_penalty = (mixed_slopes - 1.)**2
            gradient_penalty = gradient_penalty * self.d_opt['reg_interval']

            # perfrom gradient penalty
            d_loss += wgan_epsilon * gradient_penalty
            d_loss = tf.reduce_mean(d_loss)
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))  
        return d_loss, tf.reduce_mean(gradient_penalty)


    @tf.function
    def g_train_step(self, z, labels):
        with tf.GradientTape() as g_tape:
            # fw pass
            fake_images = self.generator(z, labels, training=True)
            fake_scores = self.discriminator(fake_images, labels, training=True)

            # gan loss
            g_loss = tf.math.softplus(-fake_scores)
            g_loss = tf.reduce_mean(g_loss)
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        return g_loss

    @tf.function
    def g_reg_train_step(self, z, labels):
        with tf.GradientTape() as g_tape:
            # forward pass
            fake_images, fake_dlatents = self.generator(z, labels, training=True, return_dlatents=True)
            fake_scores = self.discriminator(fake_images, labels, training=True)

            # gan loss
            g_loss = tf.math.softplus(-fake_scores)

            # path length regularization
            # fake_dlatents = self.generator.mapping_network(z, labels)
            with tf.GradientTape() as pl_tape:
                # fake_images, dlatents = self.generator(z, labels, training=True, return_dlatents=True)
                pl_tape.watch(fake_dlatents)
                fake_images_out = self.generator.synthesis_network(fake_dlatents)
                
                # Compute |J*y|.
                pl_noise = tf.random.normal(tf.shape(fake_images_out), mean=0.0, stddev=1.0, dtype=tf.float32) * self.pl_denorm
                pl_noise_added = tf.reduce_sum(fake_images_out * pl_noise)

            pl_grads = pl_tape.gradient(pl_noise_added, fake_dlatents)
            pl_lengths = tf.math.sqrt(tf.reduce_mean(tf.reduce_sum(tf.math.square(pl_grads), axis=2), axis=1))

            # Track exponential moving average of |J*y|.
            pl_mean_val = self.pl_mean + self.pl_decay * (tf.reduce_mean(pl_lengths) - self.pl_mean)
            self.pl_mean.assign(pl_mean_val)

            # Calculate (|J*y|-a)^2.
            pl_penalty = tf.square(pl_lengths - self.pl_mean)

            # compute
            pl_reg = pl_penalty * self.pl_weight

            # combine
            g_loss += pl_reg
            g_loss = tf.reduce_mean(g_loss)

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        return g_loss, tf.reduce_mean(pl_reg)

    def train(self):
        if self.reached_max_steps:
            return

        print('Start training ...')

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.ckpt_dir)

        # loss metrics
        metric_g_loss = tf.keras.metrics.Mean('g_loss', dtype=tf.float32)
        metric_d_loss = tf.keras.metrics.Mean('d_loss', dtype=tf.float32)
        metric_r1_reg = tf.keras.metrics.Mean('r1_reg', dtype=tf.float32)
        # metric_gradient_penalty = tf.keras.metrics.Mean('gradient_penalty', dtype=tf.float32)
        metric_pl_reg = tf.keras.metrics.Mean('pl_reg', dtype=tf.float32)

        # start training
        print('max steps: {}'.format(self.max_steps))
        losses = {'g_loss': 0.0, 'd_loss': 0.0, 'r1_reg': 0.0, 'pl_reg': 0.0}
        # losses = {'g_loss': 0.0, 'd_loss': 0.0, 'gradient_penalty': 0.0, 'pl_reg': 0.0}
        t_start = time.time()

        # while True:
            # real_images, labels = self.dataset_loader.get_batch()
        for real_images, labels in self.dataset_loader.train_ds:
            real_images = tf.transpose(real_images, [0, 3, 1, 2])
            z = tf.random.normal(shape=[tf.shape(real_images)[0], self.g_kwargs['z_dim']], dtype=tf.dtypes.float32)

            # get current step
            step = self.g_optimizer.iterations.numpy()

            # d train step
            if step % self.d_opt['reg_interval'] == 0:
                d_loss, r1_reg = self.d_reg_train_step(z, real_images, labels)
                # d_loss, gp = self.d_wgan_gp(z, real_images, labels)

                # update value for printing
                # losses['d_loss'] = d_loss.numpy()
                losses['r1_reg'] = r1_reg.numpy()
                # losses['gradient_penalty'] = gp.numpy()

                # update metrics
                metric_d_loss(d_loss)
                metric_r1_reg(r1_reg)
                # metric_gradient_penalty(gp)
            else:
                d_loss = self.d_train_step(z, real_images, labels)

                # update values for printing
                losses['d_loss'] = d_loss.numpy()
                losses['r1_reg'] = 0.0
                # losses['gradient_penalty'] = 0.0

                # update metrics
                metric_d_loss(d_loss)
            
            # update Gs
            self.g_clone.set_as_moving_average_of(self.generator)

            # g train step
            if step % self.g_opt['reg_interval'] == 0:
                g_loss, pl_reg = self.g_reg_train_step(z, labels)

                # update values for printing
                losses['g_loss'] = g_loss.numpy()
                losses['pl_reg'] = pl_reg.numpy()

                # update metrics
                metric_g_loss(g_loss)
                metric_pl_reg(pl_reg)
            else:
                g_loss = self.g_train_step(z, labels)

                # update values for printing
                losses['g_loss'] = g_loss.numpy()
                losses['pl_reg'] = 0.0

                # update metrics
                metric_g_loss(g_loss)

            # save to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('g_loss', metric_g_loss.result(), step=step)
                tf.summary.scalar('d_loss', metric_d_loss.result(), step=step)
                tf.summary.scalar('r1_reg', metric_r1_reg.result(), step=step)
                # tf.summary.scalar('gradient_penalty', metric_gradient_penalty.result(), step=step)
                tf.summary.scalar('pl_reg', metric_pl_reg.result(), step=step)
                tf.summary.histogram('w_avg', self.generator.w_avg, step=step)

            # save every save_step
            if step % self.save_step == 0:
                self.manager.save(checkpoint_number=step)

            # save every image_summary_step
            if step % self.image_summary_step == 0:
                # add summary img
                summary_image = self.sample_images_tensorboard(real_images, labels)
                with train_summary_writer.as_default():
                    tf.summary.image('images', summary_image, step=step)

            # print every print_step
            if step % self.print_step == 0:
                elapsed = time.time() - t_start
                print(self.log_template.format(step, elapsed,
                                               losses['d_loss'], losses['g_loss'], losses['r1_reg'], losses['pl_reg']))
                # print(self.log_template.format(step, elapsed,
                #                                losses['d_loss'], losses['g_loss'], losses['gradient_penalty'], losses['pl_reg']))                               

                # reset timer
                t_start = time.time()

            # check exit status
            if step >= self.max_steps:
                break
        
        # get current step
        step = self.g_optimizer.iterations.numpy()
        elapsed = time.time() - t_start
        print(self.log_template.format(step, elapsed,
                                               losses['d_loss'], losses['g_loss'], losses['r1_reg'], losses['pl_reg']))
        # print(self.log_template.format(step, elapsed,
        #                                        losses['d_loss'], losses['g_loss'], losses['gradient_penalty'], losses['pl_reg']))

        # save last checkpoint
        self.manager.save(checkpoint_number=step)
        return

    def sample_images_tensorboard(self, real_images, labels_in):
        reals = real_images[:self.n_samples, :, :, :]
        labels = labels_in[:self.n_samples, :]
        latents = tf.random.normal(shape=(self.n_samples, self.g_kwargs['z_dim']), dtype=tf.dtypes.float32)
        # dummy_labels = tf.ones((self.n_samples, self.g_kwargs['labels_dim']), dtype=tf.dtypes.float32)

        # run networks
        fake_images_00 = self.g_clone(latents, labels, truncation_psi=0.0, training=False)
        fake_images_05 = self.g_clone(latents, labels, truncation_psi=0.5, training=False)
        fake_images_07 = self.g_clone(latents, labels, truncation_psi=0.7, training=False)
        fake_images_10 = self.g_clone(latents, labels, truncation_psi=1.0, training=False)

        # merge on batch dimension: [5 * n_samples, 3, out_res, out_res]
        out = tf.concat([reals, fake_images_00, fake_images_05, fake_images_07, fake_images_10], axis=0)

        # prepare for image saving: [5 * n_samples, out_res, out_res, 3]
        out = postprocess_images(out)

        # resize to save disk spaces: [5 * n_samples, size, size, 3]
        size = min(self.out_res, 256)
        out = tf.image.resize(out, size=[size, size])

        # make single image and add batch dimension for tensorboard: [1, 5 * size, n_samples * size, 3]
        out = merge_batch_images(out, size, rows=5, cols=self.n_samples)
        out = np.expand_dims(out, axis=0)
        return out

def main():
    # parser
    parser = argparse.ArgumentParser(description='stylegan2')
    parser.add_argument('--model_base_dir', default='./models', type=str)
    parser.add_argument('--datasets_dir', default='./datasets/logo-color-label', type=str)
    parser.add_argument('--res', default=256, type=int)
    parser.add_argument('--shuffle_buffer_size', default=1000, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--labels_dim', default=0, type=int)
    parser.add_argument('--n_total_image', default=25000000, type=int)
    parser.add_argument('--name', default='stylegan2', type=str)
    agrs = vars(parser.parse_args())
    #--------------------
    labels_dim = agrs['labels_dim']

    train_params = EasyDict()
    train_params['model_base_dir'] = agrs['model_base_dir']
    train_params['datasets_dir'] = agrs['datasets_dir']
    train_params['shuffle_buffer_size'] = agrs['shuffle_buffer_size']
    train_params['batch_size'] = agrs['batch_size']
    train_params['n_total_image'] = agrs['n_total_image']
    train_params['n_samples'] = 4
    train_params['lazy_regularization'] = True
    train_params['name'] = agrs['name']
    train_params['max_to_keep'] = 5

    # generator args
    g_kwargs = EasyDict()
    g_kwargs['z_dim'] = 512
    g_kwargs['w_dim'] = 512
    g_kwargs['labels_dim'] = labels_dim
    g_kwargs['n_mapping'] = 8
    g_kwargs['resolution'] = 256
    g_kwargs['w_ema_decay'] = 0.995
    g_kwargs['style_mixing_prob'] = 0.9

    # discriminator args
    d_kwargs = EasyDict()
    d_kwargs['labels_dim'] = labels_dim
    d_kwargs['resolution'] = 256

    train_params['g_kwargs'] = g_kwargs
    train_params['d_kwargs'] = d_kwargs

    # optimizer
    g_opt = {
        'learning_rate': 0.002,
        'beta1': 0.0,
        'beta2': 0.99,
        'epsilon': 1e-08,
        'reg_interval': 4
    }
    d_opt = {
        'learning_rate': 0.001,
        'beta1': 0.0,
        'beta2': 0.99,
        'epsilon': 1e-08,
        'reg_interval': 16
    }

    train_params['g_opt'] = g_opt
    train_params['d_opt'] = d_opt

    # print(train_params)
    # train ------------------------------------
    trainer = Trainer(**train_params)
    trainer.train()
    return

if __name__ == '__main__':
    main()
