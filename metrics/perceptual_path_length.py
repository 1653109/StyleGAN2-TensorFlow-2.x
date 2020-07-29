import tensorflow as tf
import numpy as np
import random

from stylegan2_generator import StyleGan2Generator
from utils.utils_stylegan2 import lerp, wrap_frozen_graph

def normalize(v):
    return v / tf.sqrt(tf.reduce_sum(tf.square(v), axis=-1, keepdims=True))

def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = tf.reduce_sum(a * b, axis=-1, keepdims=True)
    p = t * tf.math.acos(d)
    c = normalize(b - d * a)
    d = a * tf.math.cos(p) + c * tf.math.sin(p)
    return normalize(d)

class PPL:
    def __init__(
        self, 
        name='perceptual_path_length', 
        epsilon=1e-4, 
        space='z',
        num_samples=5000,
        sampling='full',
        **kwargs):

        assert space in ['w', 'z']
        assert sampling in ['full', 'end']
        # super(PPL, self).__init__(name=name, **kwargs)
        self.epsilon = epsilon
        self.space = space
        self.num_samples = num_samples
        self.sampling = sampling
        # self.PPL = self.add_weight(name='ppl', initializer='zeros')

    def evaluate(self, Gs, batch_size=8, labels_dim=4):
        assert isinstance(Gs, StyleGan2Generator)
        all_distances = []
        for _ in range(0, self.num_samples, batch_size):
            lat_t01 = tf.random.normal([batch_size] + [Gs.z_dim])
            lerp_t = tf.random.uniform([batch_size // 2], 0.0, 1.0 if self.sampling == 'full' else 0.0)

            # create labels
            label_idx = random.randint(0, labels_dim - 1)
            indices = np.full(shape=(batch_size), fill_value=label_idx, dtype='int32')
            labels = tf.one_hot(indices, labels_dim, axis=-1, dtype=tf.float32)

            # Interpolate in W or Z
            if self.space == 'z':
                lat_t0, lat_t1 = lat_t01[0::2], lat_t01[1::2]
                lat_e0 = slerp(lat_t0, lat_t1, lerp_t[:, np.newaxis])
                lat_e1 = slerp(lat_t0, lat_t1, lerp_t[:, np.newaxis] + self.epsilon)
                lat_e01 = tf.reshape(tf.stack([lat_e0, lat_e1], axis=1), lat_t01.shape)
                dlat_e01 = Gs.mapping_network(lat_e01, labels)
            else: # w
                dlat_t01 = Gs.mapping_network(lat_t01, labels)
                dlat_t01 = tf.cast(dlat_t01, tf.float32)
                dlat_t0, dlat_t1 = dlat_t01[0::2], dlat_t01[1::2]
                dlat_e0 = lerp(dlat_t0, dlat_t1, lerp_t[:, np.newaxis, np.newaxis])
                dlat_e1 = lerp(dlat_t0, dlat_t1, lerp_t[:, np.newaxis, np.newaxis] + self.epsilon)
                dlat_e01 = tf.reshape(tf.stack([dlat_e0, dlat_e1], axis=1), dlat_t01.shape)

            images = Gs.synthesis_network(dlat_e01)
            images = tf.cast(images, tf.float32)

            factor = images.shape[2] // 256
            if factor > 1:
                images = tf.reshape(images, [-1, images.shape[1], images.shape[2] // factor, factor, images.shape[3] // factor, factor])
                images = tf.reduce_mean(images, axis=[3,5])

            images = (images + 1) * (255 / 2)

            img1, img2 = images[0::2], images[1::2]

            with tf.io.gfile.GFile('weights/vgg16_zhang_perceptual.pb', "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                _ = graph_def.ParseFromString(f.read())

            inputs=["vgg16_perceptual_distance/images_a:0", "vgg16_perceptual_distance/images_b:0"]
            outputs=["vgg16_perceptual_distance/Reshape_2:0"]
            vgg16 = wrap_frozen_graph(graph_def=graph_def,
                                            inputs=inputs,
                                            outputs=outputs,
                                            print_graph=False)
            kwargs = {
                'vgg16_perceptual_distance/images_a': img1,
                'vgg16_perceptual_distance/images_b': img2
            }
            result = vgg16(**kwargs)
            result = [result[0] * (1 / self.epsilon**2)]
            all_distances += result
        
        all_distances = np.concatenate(all_distances, axis=0)

        lo = np.percentile(all_distances, 1, interpolation='lower')
        hi = np.percentile(all_distances, 99, interpolation='higher')
        filtered_distances = np.extract(np.logical_and(lo <= all_distances, all_distances <= hi), all_distances)
        return np.mean(filtered_distances)


def main():
    pass

if __name__ == "__main__":
    main()
