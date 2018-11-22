from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import cv2

import numpy as np
import tensorflow as tf

_IMG_X = 303
_IMG_Y = 303
_IMG_Z = 3

def int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]

  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def image_to_tfexample(img,x,y,z):
  return tf.train.Example(features=tf.train.Features(feature={
      'img': bytes_feature(img),
      'x': int64_feature(x),
      'y': int64_feature(y),
      'z': int64_feature(z)}))


def _get_output_filename(dataset_dir, split_name):
  return '%s/record_%s.tfrecord' % (dataset_dir, split_name)


def _add_to_tfrecord(foo, num_images, tfrecord_writer):
    pos = 0
    start_time = time.time()
    
    for f in foo:
        img = cv2.imread(f)


        img = np.reshape(img, [_IMG_X*_IMG_Y*_IMG_Z])
        img = img.tobytes()
        example = image_to_tfexample(img,_IMG_X,_IMG_Y,_IMG_Z)

        tfrecord_writer.write(example.SerializeToString())
        pos += 1
        if (pos % 1 == 0):
          sys.stdout.write('\r>> Converting image %d/%d   --   Duration: %.0f s   --   %.2f percent Completed' % (pos, num_images, time.time() - start_time, 100 * (pos / num_images)))
          sys.stdout.flush()
    print('\n')



def run(dataset_dir):
  tfrecord_filename = _get_output_filename(dataset_dir,'test')

  #if tf.gfile.Exists(tfrecord_filename):
  #  print('Dataset files already exist. Exiting without re-creating them.')
  #  return

  foo = os.listdir(dataset_dir)
  foo = [os.path.join(dataset_dir,f) for f in foo if f.find('.jpg')!=-1]
  num_samples = len(foo)

  # First, process the validation data:
  with tf.python_io.TFRecordWriter(tfrecord_filename) as tfrecord_writer:
    _add_to_tfrecord(foo, num_samples, tfrecord_writer)

  print('\nFinished converting the dataset!')

run('./raw')