import tensorflow as tf
import numpy as np

from utils.weights_map import available_weights, synthesis_weights, mapping_weights, weights_stylegan2_dir
from utils.utils_stylegan2 import nf, lerp, EasyDict
from layers.dense_layer import DenseLayer
from layers.synthesis_main_layer import SynthesisMainLayer
from layers.to_rgb_layer import ToRgbLayer
from layers.label_embedding import LabelEmedding
from dnnlib.ops.upfirdn_2d import upsample_2d

class MappingNetwork(tf.keras.layers.Layer):
    """
    StyleGan2 generator mapping network, from z to dlatents for tensorflow 2.x
    """
    def __init__(
        self,
        w_dim       = 512,
        labels_dim  = 0,
        n_mapping   = 8,
        n_broadcast = 18,
        **kwargs):
        
        super(MappingNetwork, self).__init__(**kwargs)
        
        self.dlatent_size = w_dim
        self.dlatent_vector = n_broadcast
        self.mapping_layers = n_mapping
        self.lrmul = 0.01
        self.labels_dim = labels_dim

        # Embed labels
        if labels_dim > 0:
            self.labels_embedding = LabelEmedding(embed_dim=self.dlatent_size, name='LabelConcat/weight')
        
    def build(self, input_shape):
        # Embed labels and concatenate them with latents.
        # if self.labels_dim > 0:
        #     self.label_concat_weight = self.add_weight(name='LabelConcat/weight', shape=[self.labels_dim, self.dlatent_size], initializer=tf.random_normal_initializer(0, 1), trainable=True)

        self.weights_dict = {}
        for i in range(self.mapping_layers):
            setattr(self, 'Dense{}'.format(i), DenseLayer(fmaps=512, lrmul=self.lrmul, name='Dense{}'.format(i)))
    
        self.g_mapping_broadcast = tf.keras.layers.RepeatVector(self.dlatent_vector)
            
    def call(self, z, labels_in=None):
        z = tf.cast(z, 'float32')
        x = z

        # Embed labels and concatenate them with latents.
        if labels_in is not None and self.labels_dim > 0:
            # labels_in.set_shape([None, latent_size])
            assert labels_in.shape[1] == self.labels_dim
            # y = tf.linalg.matmul(labels_in, self.label_concat_weight)
            y = self.labels_embedding(labels_in)
            x = tf.concat([x, y], axis=1)
        
        # Normalize inputs
        scale = tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8)
        # x = tf.math.multiply(z, scale)
        x = tf.math.multiply(x, scale)
        
        # Mapping
        for i in range(self.mapping_layers):
        
            x = getattr(self, 'Dense{}'.format(i))(x)
            x = tf.math.multiply(tf.nn.leaky_relu(x, 0.2), tf.math.sqrt(2.))
        
        # Broadcasting
        dlatents = self.g_mapping_broadcast(x)
        
        return dlatents

class SynthesisNetwork(tf.keras.layers.Layer):
    """
    StyleGan2 generator synthesis network from dlatents to img tensor for tensorflow 2.x
    """
    def __init__(self, resolution=1024, impl='cuda', gpu=True, **kwargs):
        """
        Parameters
        ----------
        resolution : int, optional
            Resolution output of the synthesis network, will be parsed to the floor integer power of 2. 
            The default is 1024.
        impl : str, optional
            Wether to run some convolutions in custom tensorflow operations or cuda operations. 'ref' and 'cuda' available.
            The default is 'cuda'.
        gpu : boolean, optional
            Wether to use gpu. The default is True.

        """
        super(SynthesisNetwork, self).__init__(**kwargs)
        
        self.impl = impl
        self.gpu = gpu
        self.resolution = resolution
        
        self.resolution_log2 = int(np.log2(self.resolution))
        self.resample_kernel = [1, 3, 3, 1]
        
    def build(self, input_shape):
        
        #constant layer
        self.const_4_4 = self.add_weight(name='4x4/Const/const', shape=(1, 512, 4, 4), 
                                        initializer=tf.random_normal_initializer(0, 1), trainable=True)
        #early layer 4x4
        self.layer_4_4 = SynthesisMainLayer(fmaps=nf(1), impl=self.impl, gpu=self.gpu, name='4x4')
        self.torgb_4_4 = ToRgbLayer(impl=self.impl, gpu=self.gpu, name='4x4')
        #main layers
        for res in range(3, self.resolution_log2 + 1):
            res_str = str(2**res)
            setattr(self, 'layer_{}_{}_up'.format(res_str, res_str), 
                    SynthesisMainLayer(fmaps=nf(res-1), impl=self.impl, gpu=self.gpu, up=True, name='{}x{}'.format(res_str, res_str)))
            setattr(self, 'layer_{}_{}'.format(res_str, res_str), 
                    SynthesisMainLayer(fmaps=nf(res-1), impl=self.impl, gpu=self.gpu, name='{}x{}'.format(res_str, res_str)))
            setattr(self, 'torgb_{}_{}'.format(res_str, res_str), 
                    ToRgbLayer(impl=self.impl, gpu=self.gpu, name='{}x{}'.format(res_str, res_str)))
        
    def call(self, dlatents_in):
        
        dlatents_in = tf.cast(dlatents_in, 'float32')
        y = None
        
        # Early layers
        x = tf.tile(tf.cast(self.const_4_4, 'float32'), [tf.shape(dlatents_in)[0], 1, 1, 1])
        x = self.layer_4_4(x, dlatents_in[:, 0])
        y = self.torgb_4_4(x, dlatents_in[:, 1], y)
                
        # Main layers
        for res in range(3, self.resolution_log2 + 1):
            x = getattr(self, 'layer_{}_{}_up'.format(2**res, 2**res))(x, dlatents_in[:, res*2-5])
            x = getattr(self, 'layer_{}_{}'.format(2**res, 2**res))(x, dlatents_in[:, res*2-4])
            y = upsample_2d(y, k=self.resample_kernel, impl=self.impl, gpu=self.gpu)
            y = getattr(self, 'torgb_{}_{}'.format(2**res, 2**res))(x, dlatents_in[:, res*2-3], y)

        images_out = y
        return tf.identity(images_out, name='images_out')
    
class StyleGan2Generator(tf.keras.layers.Layer):
    """
    StyleGan2 generator config f for tensorflow 2.x
    """
    def __init__(
        self, 
        resolution          = 1024, 
        weights             = None, 
        impl                = 'cuda', 
        gpu                 = True, 
        config              = "f",
        labels_dim          = 0,
        z_dim               = 512,
        w_dim               = 512,
        n_mapping           = 8,
        w_ema_decay         = 0.995,
        style_mixing_prob   = 0.9,
        truncation_psi      = 0.5,
        truncation_cutoff   = None,
        **kwargs):
        """
        Parameters
        ----------
        resolution : int, optional
            Resolution output of the synthesis network, will be parsed 
            to the floor integer power of 2. 
            The default is 1024.
        weights : string, optional
            weights name in weights dir to be loaded. The default is None.
        impl : str, optional
            Wether to run some convolutions in custom tensorflow operations 
            or cuda operations. 'ref' and 'cuda' available.
            The default is 'cuda'.
        gpu : boolean, optional
            Wether to use gpu. The default is True.

        """
        super(StyleGan2Generator, self).__init__(**kwargs)
        
        self.resolution = resolution
        if weights is not None: self.__adjust_resolution(weights)

        self.z_dim = z_dim
        self.w_dim = w_dim
        self.labels_dim = labels_dim
        self.n_mapping = n_mapping
        self.w_ema_decay = w_ema_decay
        self.style_mixing_prob = style_mixing_prob

        self.n_broadcast = (int(np.log2(self.resolution)) - 1)*2
        self.mixing_layer_indices = np.arange(self.n_broadcast)[np.newaxis, :, np.newaxis]

        self.mapping_network = MappingNetwork(name='Mapping_network', labels_dim=self.labels_dim, n_broadcast=self.n_broadcast)
        self.synthesis_network = SynthesisNetwork(resolution=self.resolution, impl=impl, 
                                                  gpu=gpu, name='Synthesis_network')
        
        # load weights
        if weights is not None:
            #we run the network to define it, not the most efficient thing to do...
            labels = tf.zeros(shape=[1, self.labels_dim])
            _ = self(tf.zeros(shape=(1, 512)), labels_in=labels)
            self.__load_weights(weights)

    def build(self, input_shape):
        # w_avg
        self.w_avg = tf.Variable(tf.zeros(shape=[self.w_dim], dtype=tf.dtypes.float32), name='w_avg', trainable=False)

    def update_moving_average_of_w(self, dlatents):
        # compute avarage of current w
        batch_avg = tf.reduce_mean(dlatents[:, 0], axis=0)
        # compute moving average of w and update*assign) a_avg
        self.w_avg.assign(lerp(batch_avg, self.w_avg, self.w_ema_decay))
        return

    def set_as_moving_average_of(self, src_net, beta=0.99, beta_nontrainable=0.0):
        '''
        coming soon
        '''
        def split_first_name(name):
            splitted = name.split('/')
            new_name = '/'.join(splitted[1:])
            return new_name
        
        for cw in self.trainable_weights:
            cw_name = split_first_name(cw.name)
            for sw in src_net.trainable_weights:
                sw_name = split_first_name(sw.name)
                if cw_name == sw_name:
                    assert sw.shape == cw.shape
                    cw.assign(lerp(sw, cw, beta))
                    break

        for cw in self.non_trainable_weights:
            cw_name = split_first_name(cw.name)
            for sw in src_net.non_trainable_weights:
                sw_name = split_first_name(sw.name)
                if cw_name == sw_name:
                    assert sw.shape == cw.shape
                    cw.assign(lerp(sw, cw, beta_nontrainable))
                    break
        return

    def style_mixing_regularization(self, z1, labels, w1):
        '''
        get another w and broadcast it
        '''
        z2 = tf.random.normal(shape=tf.shape(z1), dtype='float32')
        w2 = self.mapping_network(z2, labels)

        # find mixing limit index
        if tf.random.uniform([], 0.0, 1.0) < self.style_mixing_prob:
            mixing_cutoff_index = tf.random.uniform([], 1, self.n_broadcast, dtype='int32')
        else:
            mixing_cutoff_index = tf.constant(self.n_broadcast, dtype='int32')

        # mix it
        mixed_w = tf.where(
            condition=tf.broadcast_to(self.mixing_layer_indices < mixing_cutoff_index, tf.shape(w1)),
            x=w1,
            y=w2
        )
        return mixed_w

    def truncation_trick(self, w, truncation_cutoff, truncation_psi):
        ones = np.ones_like(self.mixing_layer_indices, dtype=np.float32)
        if truncation_cutoff is None:
            truncation_coefs = ones * truncation_psi
        else:
            truncation_coefs = ones
            for idx in range(self.n_broadcast):
                if idx < truncation_cutoff:
                    truncation_coefs[:, idx, :] = truncation_psi

        truncated_w = lerp(self.w_avg, w, truncation_coefs)
        return truncated_w

    @tf.function
    def call(self, z, labels_in=None, truncation_cutoff=None, truncation_psi=1.0, training=None, mask=None, return_dlatents=False):
        """

        Parameters
        ----------
        z : tensor, latent vector of shape [batch, 512]

        Returns
        -------
        img : tensor, image generated by the generator of shape  [batch, channel, height, width]

        """
        dlatents = self.mapping_network(z, labels_in)

        if training:
            self.update_moving_average_of_w(dlatents)
            dlatents = self.style_mixing_regularization(z, labels_in, dlatents)
        else:
            dlatents = self.truncation_trick(dlatents, truncation_cutoff, truncation_psi)

        img = self.synthesis_network(dlatents)

        if return_dlatents:
            return img, dlatents
        return img
    
    def __adjust_resolution(self, weights_name):
        """
        Adjust resolution of the synthesis network output. 
        
        Parameters
        ----------
        weights_name : name of the weights

        Returns
        -------
        None.

        """
        if  weights_name == 'ffhq': 
            self.resolution = 1024
        elif weights_name == 'car': 
            self.resolution = 512
        elif weights_name in ['cat', 'church', 'horse', 'logo']: 
            self.resolution = 256
    
    def __load_weights(self, weights_name):
        """
        Load pretrained weights, stored as a dict with numpy arrays.
        Parameters
        ----------
        weights_name : name of the weights

        Returns
        -------
        None.

        """
        
        if (weights_name in available_weights) and type(weights_name) == str:
            data = np.load(weights_stylegan2_dir + weights_name + '.npy', allow_pickle=True)[()]
            
            weights_mapping = [data.get(key) for key in mapping_weights]
            weights_synthesis = [data.get(key) for key in synthesis_weights[weights_name]]
            
            self.mapping_network.set_weights(weights_mapping)
            self.synthesis_network.set_weights(weights_synthesis)
            
            print("Loaded {} generator weights!".format(weights_name))
        else:
            raise Exception('Cannot load {} weights'.format(weights_name))

# def main():
#     g_kwargs = EasyDict()
#     g_kwargs['impl'] = 'cuda'
#     g_kwargs['gpu'] = True
#     g_kwargs['labels_dim'] = 19
#     g_kwargs['weights'] = 'logo'

#     generator = StyleGan2Generator(**g_kwargs)

#     seed = 96
#     rnd = np.random.RandomState(seed)
#     z = rnd.randn(1, 512).astype('float32')

# if __name__ == '__main__':
#     main()
