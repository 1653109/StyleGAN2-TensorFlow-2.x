#
# from: https://www.tensorflow.org/tutorials/load_data/images
#

import tensorflow as tf
import pathlib
import os
import numpy as np

class DatasetLoader:
    """ Helper for load the dataset using tf.Dataset class """
    def __init__(self, path_dir, resolution, batch_size, shuffle_buffer_size=100, cache_file=True):
        """
        path_dir : Directory of the image dataset
        resolution : Resolution of the training images
        batch_size : Batch size
        cache_file : filepath to store cache files .tfcache extension
        """
        self.resolution = resolution
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.shuffle_buffer_size = shuffle_buffer_size
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        
        # list_ds = tf.data.Dataset.list_files(str(path_dir+'/*'))
        data_dir = pathlib.Path(path_dir)
        list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
        self.CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])

        labeled_ds = list_ds.map(self.process_path, num_parallel_calls=self.AUTOTUNE)
        self.train_ds = self.prepare_for_training(labeled_ds, cache=self.cache_file)

    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [self.resolution, self.resolution])

    def process_path(self, file_path):
        label = self._get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        return self.decode_img(img), label

    def _get_label(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        label = parts[-2] == self.CLASS_NAMES
        label = tf.cast(label, dtype='float32')
        return label
    
    def get_batch(self):
        images, label = next(iter(self.train_ds))
        return tf.transpose(images, [0, 3, 1, 2]), label 

    def prepare_for_training(self, ds, cache=True):
      # use `.cache(filename)` to cache preprocessing work for datasets that don't
      # fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        ds = ds.shuffle(buffer_size=self.shuffle_buffer_size)
        # Repeat forever
        ds = ds.repeat()
        ds = ds.batch(self.batch_size)
        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds
