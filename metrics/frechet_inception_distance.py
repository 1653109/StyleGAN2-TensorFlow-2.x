import tensorflow as tf
import numpy as np
import os
import pickle
import random
import scipy

from stylegan2_generator import StyleGan2Generator
from utils.dataset_loader import DatasetLoader
from utils.utils_stylegan2 import wrap_frozen_graph, convert_images_to_uint8

class FID:
    def __init__(self, name="FID", n_images=3000, dataset_loader=None, **kwargs):
        assert isinstance(dataset_loader, DatasetLoader)
        self.n_images = n_images
        self.dataset_loader = dataset_loader
        self.name = name

    def evaluate(self, Gs, batch_size=8, labels_dim=4):
        assert isinstance(Gs, StyleGan2Generator)
        # load network
        with tf.io.gfile.GFile('weights/vgg16_zhang_perceptual.pb', "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            _ = graph_def.ParseFromString(f.read())

        inception = wrap_frozen_graph(graph_def=graph_def,
                                        inputs=['InceptionV3/images_in:0'],
                                        outputs=['InceptionV3/Reshape:0'],
                                        print_graph=False)

        activations = np.empty([self.n_images, inception.outputs[0].shape[1]], dtype=np.float32)

        # calc statistics for reals
        cache_file = self._get_cache_file_for_reals()
        os.makedirs(cache_file, exist_ok=True)
        if os.path.isfile(cache_file):
            mu_real, sigma_real = pickle.load(open(cache_file, 'rb'))
        else:
            for idx, images in enumerate(self._iterate_reals()):
                begin = idx * batch_size
                end = min(begin + batch_size, self.n_images)
                inception_args = {'InceptionV3/images_in': images[:end-begin]}
                activations[begin:end] = inception(**inception_args)
                if end == self.n_images:
                    break
            mu_real = np.mean(activations, axis=0)
            sigma_real = np.cov(activations, rowvar=False)
            with open(cache_file, 'wb') as f:
                pickle.dump((mu_real, sigma_real), f, protocol=pickle.HIGHEST_PROTOCOL)

        # calc statistics for fakes
        for begin in range(0, self.n_images, batch_size):
            end = min(begin + batch_size, self.n_images)
            # produce fakes
            latents = tf.random.normal([batch_size] + [Gs.z_dim])
            label_idx = random.randint(0, labels_dim - 1)
            indices = np.full(shape=(batch_size), fill_value=label_idx, dtype='int32')
            labels = tf.one_hot(indices, labels_dim, axis=-1, dtype=tf.float32)
            images = Gs(latents, labels, training=False)
            images = convert_images_to_uint8(images)
            inception_args = {'InceptionV3/images_in': images}
            activations[begin:end] = np.concatenate(inception(**inception_args), axis=0)[:end-begin]
        mu_fake = np.mean(activations, axis=0)
        sigma_fake = np.cov(activations, rowvar=False)

        # calc FID
        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False)
        dist = m + np.trace(sigma_fake + sigma_real - 2*s)
        return np.real(dist)

        
    def _get_cache_file_for_reals(self, extension='pkl'):
        dataset_name = self.dataset_loader.path_dir.split('/')[-1]
        return os.path.join('.stylegan2-cache', '{}-{}-{}.{}'.format(self.name, dataset_name, self.n_images, extension))
        
    def _iterate_reals(self):
        while True:
            images, _labels = self.dataset_loader.get_batch()
            yield images

