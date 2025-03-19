# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains utilities for downloading and converting datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
from PIL import Image
try:
  from StringIO import StringIO
except ImportError:
  from io import StringIO

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
#import boto.s3.connection

'''
access_key = "M6AAMICSWJ4W13PTDQTE"
secret_key = "io9mvKlzWCFlSUhTVKdleXVi9JVXBaNKXYoFil4O"
rgw_host = "10.129.135.110"

conn = boto.connect_s3(
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    host=rgw_host,
    is_secure=False,
    calling_format=boto.s3.connection.OrdinaryCallingFormat(),
    )
'''

slim = tf.contrib.slim

LABELS_FILENAME = 'labels.txt'


def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    a TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format, height, width, class_id, file_name='temp.jpg', label_name='', is_crop="0"):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/class/labelname': bytes_feature(label_name),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
      'image/filename': bytes_feature(file_name),
      'image/iscrop': int64_feature(int(is_crop)),
  }))


def file_to_tfexample(filepath, image_format, height, width, class_id, file_name='temp.jpg', label_name=''):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/file': bytes_feature(filepath),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/class/labelname': bytes_feature(label_name),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
      'image/filename': bytes_feature(file_name),
  }))


def rgw_to_tfexample(bucket, key, image_format, height, width, class_id, file_name='temp.jpg', label_name=''):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/rgw_bucket': bytes_feature(bucket),
      'image/rgw_key': bytes_feature(key),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/class/labelname': bytes_feature(label_name),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
      'image/filename': bytes_feature(file_name),
  }))


def download_and_uncompress_tarball(tarball_url, dataset_dir):
  """Downloads the `tarball_url` and uncompresses it locally.

  Args:
    tarball_url: The URL of a tarball file.
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = tarball_url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()
  filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
  print()
  statinfo = os.stat(filepath)
  print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
  """Writes a file with the list of class names.

  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'w') as f:
    for label in labels_to_class_names:
      class_name = labels_to_class_names[label]
      f.write('%d:%s\n' % (label, class_name))


def has_labels(dataset_dir, filename=LABELS_FILENAME):
  """Specifies whether or not the dataset directory contains a label map file.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    `True` if the labels file exists and `False` otherwise.
  """
  return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
  """Reads the labels file and returns a mapping from ID to class name.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    A map from a label (integer) to class name.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  print(">>>>>>>>>>>>>>>>>>>>>filename:",filename)
  with tf.gfile.Open(labels_filename, 'r') as f:
    lines = f.read().decode('utf-8')
  lines = lines.split('\n')
  lines = filter(None, lines)

  labels_to_class_names = {}
  for line in lines:
    index = line.index(':')
    labels_to_class_names[int(line[:index])] = line[index+1:]
  return labels_to_class_names


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)

    def read_image_dims(self, sess, image_name, image_data):
        image_name = image_name.lower()
        if image_name.find('jpg') >= 0 or image_name.find('jpeg') >= 0:
            image = self.decode_jpeg(sess, image_data)
            return 'jpg', image.shape[0], image.shape[1]
        elif image_name.find('png') >= 0:
            image = self.decode_png(sess, image_data)
            return 'png', image.shape[0], image.shape[1]
        else:
            raise ValueError('get image file not jpg or png [%s]' % image_name)

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def decode_png(self, sess, image_data):
        image = sess.run(self._decode_png,
                         feed_dict={self._decode_png_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


# base on slim.tfexample_decoder.Image
# change image_key to file_key
# change file_buf to file_name
class ImageFile(slim.tfexample_decoder.ItemHandler):
    """An ItemHandler that decodes a parsed Tensor as an image."""

    def __init__(self, file_key=None, format_key=None, shape=None,
               channels=3):
        """Initializes the image.

        Args:
          image_key: the name of the TF-Example feature in which the encoded image
            is stored.
          format_key: the name of the TF-Example feature in which the image format
            is stored.
          shape: the output shape of the image. If provided, the image is reshaped
            accordingly. If left as None, no reshaping is done. A shape should be
            supplied only if all the stored images have the same shape.
          channels: the number of channels in the image.
        """
        if not file_key:
            file_key = 'image/file'
        if not format_key:
            format_key = 'image/format'

        super(ImageFile, self).__init__([file_key, format_key])
        self._file_key = file_key
        self._format_key = format_key
        self._shape = shape
        self._channels = channels

    def tensors_to_item(self, keys_to_tensors):
        """See base class."""
        image_file = keys_to_tensors[self._file_key]
        image_format = keys_to_tensors[self._format_key]
        image_buffer = tf.read_file(image_file)

        image = self._decode(image_buffer, image_format)
        if self._shape is not None:
            image = array_ops.reshape(image, self._shape)
        return image

    def _decode(self, image_buffer, image_format):
        """Decodes the image buffer.

        Args:
          image_buffer: T tensor representing the encoded image tensor.
          image_format: The image format for the image in `image_buffer`.

        Returns:
          A decoder image.
        """
        def decode_png():
            return image_ops.decode_png(image_buffer, self._channels)
        def decode_raw():
            return parsing_ops.decode_raw(image_buffer, dtypes.uint8)
        def decode_jpg():
            return image_ops.decode_jpeg(image_buffer, self._channels)

        image = control_flow_ops.case({
            math_ops.logical_or(math_ops.equal(image_format, 'png'),
                                math_ops.equal(image_format, 'PNG')): decode_png,
            math_ops.logical_or(math_ops.equal(image_format, 'raw'),
                                math_ops.equal(image_format, 'RAW')): decode_raw,
        }, default=decode_jpg, exclusive=True)

        image.set_shape([None, None, self._channels])
        if self._shape is not None:
            image = array_ops.reshape(image, self._shape)

        return image


# base on slim.tfexample_decoder.Image
# change image_key to file_key
# change file_buf to file_name
class ImageRGW(slim.tfexample_decoder.ItemHandler):
    """An ItemHandler that decodes a parsed Tensor as an image."""

    def __init__(self, rgw_bucket_key=None, rgw_key_key=None, format_key=None, shape=None,
               channels=3):
        """Initializes the image.

        Args:
          image_key: the name of the TF-Example feature in which the encoded image
            is stored.
          format_key: the name of the TF-Example feature in which the image format
            is stored.
          shape: the output shape of the image. If provided, the image is reshaped
            accordingly. If left as None, no reshaping is done. A shape should be
            supplied only if all the stored images have the same shape.
          channels: the number of channels in the image.
        """
        if not rgw_bucket_key:
            rgw_bucket_key = 'image/rgw_bucket'
        if not rgw_key_key:
            rgw_key_key = 'image/rgw_key'
        if not format_key:
            format_key = 'image/format'

        super(ImageRGW, self).__init__([rgw_bucket_key, rgw_key_key, format_key])
        self._rgw_bucket = rgw_bucket_key
        self._rgw_key = rgw_key_key
        self._format_key = format_key
        self._shape = shape
        self._channels = channels

    def tensors_to_item(self, keys_to_tensors):
        """See base class."""
        #image_file = keys_to_tensors[self._file_key]
        image_format = keys_to_tensors[self._format_key]
        rgw_bucket = keys_to_tensors[self._rgw_bucket]
        rgw_key = keys_to_tensors[self._rgw_key]
        tf.logging.info("self._rgw_bucket: %s", self._rgw_bucket)
        tf.logging.info("self._rgw_key: %s", self._rgw_key)
        tf.logging.info("rgw_bucket: %s", rgw_bucket)
        tf.logging.info("rgw_key: %s", rgw_key)
        #image_buffer = tf.read_file(image_file)
        #bucket = conn.get_bucket(self._rgw_bucket)
        #k = bucket.get_key(self._rgw_key )
        #img_data = k.get_contents_as_string()
        img_data = tf.gfile.FastGFile("/data_old/ffwang/SZBWG/@shenzhenbowuguan/imgs/30000182/30000182_2341f713c5.JPG", 'r').read()
        tf.logging.info("len(img_data): %d", len(img_data))

        image_buffer = tf.convert_to_tensor(img_data)



        image = self._decode(image_buffer, image_format)
        if self._shape is not None:
            image = array_ops.reshape(image, self._shape)
        return image

    def _decode(self, image_buffer, image_format):
        """Decodes the image buffer.

        Args:
          image_buffer: T tensor representing the encoded image tensor.
          image_format: The image format for the image in `image_buffer`.

        Returns:
          A decoder image.
        """
        def decode_png():
            return image_ops.decode_png(image_buffer, self._channels)
        def decode_raw():
            return parsing_ops.decode_raw(image_buffer, dtypes.uint8)
        def decode_jpg():
            return image_ops.decode_jpeg(image_buffer, self._channels)

        image = control_flow_ops.case({
            math_ops.logical_or(math_ops.equal(image_format, 'png'),
                                math_ops.equal(image_format, 'PNG')): decode_png,
            math_ops.logical_or(math_ops.equal(image_format, 'raw'),
                                math_ops.equal(image_format, 'RAW')): decode_raw,
        }, default=decode_jpg, exclusive=True)

        image.set_shape([None, None, self._channels])
        if self._shape is not None:
            image = array_ops.reshape(image, self._shape)

        return image


def get_jpg_image(filename, min_size=200):
    temp_file = StringIO.StringIO()
    weight = 0
    height = 0
    try:
        if not os.path.isfile(filename):
            print("===================Warning!!! %s is not a file" % filename)
            ret = -3
        else:
            image = Image.open(filename)
            mode = image.mode

            if mode != "RGB":
                image = image.convert("RGB")

            size = image.size
            if size[0] > 2000 and size[1] > 2000:
                new_size = (int(size[0] / 2), int(size[1] / 2))
                image = image.resize(new_size)

            weight = image.size[0]
            height = image.size[1]

            if weight>min_size and height>min_size:
                image.save(temp_file, format="JPEG")
                ret = 0
            else:
                print("===================Warning!!! image too small %s, width=%d, height=%d"
                      % (filename, weight, height))
                ret = -1
    except IOError:
        print("===================Warning!!! cann't convert %s" % filename)
        ret = -2

    return ret, temp_file, weight, height

