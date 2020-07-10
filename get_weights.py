#
# from: https://github.com/rosasalberto/StyleGAN2-TensorFlow-2.x/issues/2
#

import pickle
import full_dnnlib as dnnlib
import full_dnnlib.tflib

model_path = "weights/network-snapshot-006048.pkl"

full_dnnlib.tflib.init_tf()
_G, D, Gs = pickle.load(open(model_path, "rb"))

data = {}

for key in D.trainables.keys():
    data['disc_' + key] = D.get_var(key)

for key in Gs.trainables.keys():
    data[]

