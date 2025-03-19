# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = '%s-*'

#SPLITS_TO_SIZES = {
#    'train': 398854,
#    'val': 3773,
#}
SPLITS_TO_SIZES = {
    # 2018-12-26
    #'train': 460099,
    #'train': 460861,
    # 2019-01-17
    #'train': 458900,
    #'val': 4116,
    # 2019-01-22
    #'train': 455922,
    #'val': 4116
    # 2019-01-26
    'train': 458242,
    'val': 4116
}

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A float list with a value of 0 or 1, which length is 34',
}

_NUM_CLASSES = 1

def get_split(split_name, dataset_dir, file_pattern=None, reader=None,
              train_image_count=None, val_image_count=None, label_count=None):
  """Gets a dataset tuple with instructions for reading scene.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/val split.
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  class_num = _NUM_CLASSES
  if label_count:
    class_num = label_count

  keys_to_features = {
      'image/encoded1': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/encoded2': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format1': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/format2': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
  }

  items_to_handlers = {
      'image1': slim.tfexample_decoder.Image(image_key='image/encoded1', format_key='image/format1'),
      'image2': slim.tfexample_decoder.Image(image_key='image/encoded2', format_key='image/format2'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  sample_num = SPLITS_TO_SIZES[split_name]
  if train_image_count and split_name == 'train':
      sample_num = train_image_count
  if val_image_count and split_name == 'validation':
      sample_num = val_image_count

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=sample_num,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=class_num,
      labels_to_names=labels_to_names)
