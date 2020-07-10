import numpy as np
import tensorflow as tf

class LabelEmedding(tf.keras.layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super(LabelEmedding, self).__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        weight_shape = [input_shape[1], self.embed_dim]
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=1.0)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def call(self, inputs):
        x = tf.matmul(inputs, self.w)
        return x
