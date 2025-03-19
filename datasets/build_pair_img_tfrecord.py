# -*- coding: utf-8 -*-

"""
根据图片pair列表文件，生成对应的TFrecord
输入文件第一列和第二列为图片完整路径

CUDA_VISIBLE_DEVICES="" python /home/nas01/grp_IMRECOG/wangfei11/Code/slim_scene/datasets/build_pair_img_tfrecord.py \
--train_labels_file=/home/work/wangfei11/data/CPC_Auto_Crop/CPCDataset/result_pairs_imgs.txt \
--output_directory=/home/work/wangfei11/data/CPC_Auto_Crop/TF_Pair_CPC_59w_Shuffled
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import imghdr
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

_RANDOM_SEED = 0

tf.app.flags.DEFINE_string('output_directory',
                           None,
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 16,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 8,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')

# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   n01440764
#   n01443537
#   n01484850
# where each line corresponds to a label expressed as a synset. We map
# each synset contained in the file to an integer (based on the alphabetical
# ordering). See below for details.
tf.app.flags.DEFINE_string('train_labels_file',
                           None,
                           'Train labels file')

tf.app.flags.DEFINE_string('val_labels_file',
                           None,
                           'Validation labels file')


FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image_buffer1, height1, width1, image_buffer2, height2, width2):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: string, identifier for the ground truth for the network
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height1': _int64_feature(height1),
      'image/width1': _int64_feature(width1),
      'image/format1': _bytes_feature(image_format),
      'image/encoded1': _bytes_feature(image_buffer1),
      'image/height2': _int64_feature(height2),
      'image/width2': _int64_feature(width2),
      'image/format2': _bytes_feature(image_format),
      'image/encoded2': _bytes_feature(image_buffer2)}))
  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that converts BMP to JPEG data.
    self._bmp_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_bmp(self._bmp_data, channels=3)
    self._bmp_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def bmp_to_jpeg(self, image_data):
    return self._sess.run(self._bmp_to_jpeg,
                          feed_dict={self._bmp_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image



def _is_png(filename):
  """Determine if a file contains a PNG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a PNG.
  """
  return imghdr.what(filename) == 'png'


def _is_bmp(filename):
  """Determine if a file contains a BMP format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a BMP.
  """
  return imghdr.what(filename) == 'bmp'


def _process_image(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  image_data = tf.gfile.FastGFile(filename, 'r').read()

  if _is_png(filename):
    #print('Converting PNG to JPEG for %s' % filename)
    image_data = coder.png_to_jpeg(image_data)
  elif _is_bmp(filename):
    #print('Converting BMP to JPEG for %s' % filename)
    image_data = coder.bmp_to_jpeg(image_data)

  # Decode the RGB JPEG.
  try:
    image = coder.decode_jpeg(image_data)
  except:
    print('Failed to decode image: %s' % filename)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               num_shards):
  """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in xrange(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename_pair = filenames[i]

      image_buffer1, height1, width1 = _process_image(filename_pair[0], coder)
      image_buffer2, height2, width2 = _process_image(filename_pair[1], coder)

      example = _convert_to_example(image_buffer1, height1, width1, image_buffer2, height2, width2)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(name, filenames, num_shards):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in xrange(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i+1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in xrange(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _find_image_files(labels_file):
  """Build a list of all images files and labels in the data set.

  Args:
    data_dir: string, path to the root directory of images.
    labels_file: string, path to the labels file.

  Returns:
    filenames: list of strings; each string is a path to an image file.
    labels: list of integer; each integer identifies the ground truth.
  """
  print('Determining list of input files and labels from %s.' % labels_file)
  lines = tf.gfile.FastGFile(labels_file, 'r').readlines()
  filenames = []
  for line in lines:
    sep = line.split()
    if len(sep) != 2:
      print("Error line: %s  sep(%d VS %d)" % (line, len(sep), 2))
      continue
    filenames.append((sep[0], sep[1]))

  random.seed(_RANDOM_SEED)
  random.shuffle(filenames)

  return filenames


def _process_dataset(name, labels_file, num_shards):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
  """
  filenames = _find_image_files(labels_file)
  _process_image_files(name, filenames, num_shards)


def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
  assert FLAGS.output_directory, (
      'Please specify FLAGS.output_directory')
  print('Saving results to %s' % FLAGS.output_directory)

  if not os.path.isdir(FLAGS.output_directory):
      os.mkdir(FLAGS.output_directory)

  # Run it!
  if FLAGS.val_labels_file:
    _process_dataset('val', FLAGS.val_labels_file, FLAGS.validation_shards)
  if FLAGS.train_labels_file:
    _process_dataset('train', FLAGS.train_labels_file, FLAGS.train_shards)


if __name__ == '__main__':
  tf.app.run()
