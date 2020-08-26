import os
import time
import base64
import PIL.Image
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, Response, request
from flask_cors import CORS, cross_origin
from werkzeug.exceptions import HTTPException
from werkzeug.exceptions import default_exceptions
from io import BytesIO

from utils.dataset_loader import DatasetLoader
from utils.utils_stylegan2 import postprocess_images, EasyDict, merge_batch_images, lerp
from utils.singleton import Singleton
from utils import latent_code
from stylegan2_generator import StyleGan2Generator
from stylegan2_discriminator import StyleGan2Discriminator

@Singleton
class StyleGAN2Model():
    '''
    mangaing stylegan2 model generator
    '''
    def __init__(self, ckpt_dir='models/stylegan2-logo-color-label-r1', model_name='stylegan2-logo-color-label', labels_dim=4):
        print('Initializing model ...')
        g_kwargs = EasyDict()
        g_kwargs['z_dim'] = 512
        g_kwargs['w_dim'] = 512
        g_kwargs['labels_dim'] = labels_dim
        g_kwargs['n_mapping'] = 8
        g_kwargs['resolution'] = 256
        g_kwargs['w_ema_decay'] = 0.995
        g_kwargs['style_mixing_prob'] = 0.9

        d_kwargs = EasyDict()
        d_kwargs['labels_dim'] = labels_dim
        d_kwargs['resolution'] = 256

        self.labels_dim = labels_dim
        self.model_name = model_name

        self.G = StyleGan2Generator(**g_kwargs)
        self.D = StyleGan2Discriminator(**d_kwargs)
        self.Gs = StyleGan2Generator(**g_kwargs)

        test_latent = np.ones((1, g_kwargs['z_dim']), dtype=np.float32)
        test_labels = np.ones((1, g_kwargs['labels_dim']), dtype=np.float32)
        test_images = np.ones((1, 3, 256, 256), dtype=np.float32)
        __ = self.G(test_latent, test_labels, training=False)
        _ = self.D(test_images, test_labels, training=False)
        __ = self.Gs(test_latent, test_labels, training=False)

        self.Gs.set_weights(self.G.get_weights())

        print('Loading checkpoint ...')
        ckpt = tf.train.Checkpoint(discriminator=self.D, generator=self.G, g_clone=self.Gs)
        ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt, directory=ckpt_dir, max_to_keep=5)
        ckpt.restore(ckpt_manager.latest_checkpoint)
        if ckpt_manager.latest_checkpoint:
            print('OK!')
        else:
            print('Cannot restore ckpt!')
            return
    
    def generate_from_z(self, z, label=0, psi=0.5):
        # assert 
        assert label < self.labels_dim
        n_samples = 1
        labels = np.zeros((n_samples, self.labels_dim), dtype='float32')
        labels[:, label] = 1.0
        images = self.Gs(z, labels, truncation_psi=psi, training=False)
        images = postprocess_images(images)
        images = images.numpy()
        img = PIL.Image.fromarray(images[0])
        return img

    # def generate_from_w(self, )

    def get_info(self):
        return dict(
            model = self.model_name,
            latents_dimensions = self.Gs.z_dim,
            image_shape = self.Gs.resolution,
            synthesis_input_shape = self.Gs.w_dim
        )

app = Flask(__name__)
CORS(app)

@app.errorhandler(Exception)
def handle_error(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    return jsonify(error=str(e)), code

app.config['BUNDLE_ERRORS'] = True

for ex in default_exceptions:
    app.register_error_handler(ex, handle_error)

@app.route('/api/info', methods=['GET'])
def info():
    model = StyleGAN2Model.Instance()
    return jsonify(model.get_info())

@app.route('/api/image', methods=['GET'])
def image():
    # get params
    z_str = request.args.get('z')
    psi = float(request.args.get('psi', default=1.0))
    label = int(request.args.get('label', default=0))

    if z_str is None:
        z = tf.random.normal(shape=[1, 512], dtype=tf.dtypes.float32)
    else:
        z = latent_code.decodeFloat32(z_str)
        z = z[np.newaxis, :]

    model = StyleGAN2Model.Instance()
    img = model.generate_from_z(z=z, label=label, psi=psi)

    fp = BytesIO()
    img.save(fp, PIL.Image.registered_extensions()['.png'])
    img_base64 = base64.b64encode(fp.getvalue()).decode('utf-8')

    res = {
        'image_str': 'data:image/png;base64,{}'.format(img_base64)
    }

    return jsonify(res)
    # return Response(fp.getvalue(), mimetype = 'image/png')

if __name__ == '__main__':
    # model = StyleGAN2Model()
    # z = tf.random.normal(shape=[1, 512], dtype=tf.dtypes.float32)
    # img = model.generate_from_z(z=z, label=1)
    # print(img.shape)

    app.run(debug=True)