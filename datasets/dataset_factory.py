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
"""A factory-pattern class which returns classification image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from datasets import scene_singlelabel
from datasets import scene_multilabel
from datasets import default
from datasets import default_min_crop
from datasets import pair_rank_dataset


datasets_map = {
    'scene_singlelabel': scene_singlelabel,
    'scene_multilabel': scene_multilabel,
    'default_min_crop': default_min_crop,
    'pair_rank_dataset': pair_rank_dataset,
}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None,
                train_image_count=None, val_image_count=None, label_count=None):
  """Given a dataset name and a split_name returns a Dataset.

  Args:
    name: String, the name of the dataset.
    split_name: A train/test split name.
    dataset_dir: The directory where the dataset files are stored.
    file_pattern: The file pattern to use for matching the dataset source files.
    reader: The subclass of tf.ReaderBase. If left as `None`, then the default
      reader defined by each dataset is used.

  Returns:
    A `Dataset` class.

  Raises:
    ValueError: If the dataset `name` is unknown.
  """
  #if name not in datasets_map:
  #  raise ValueError('Name of dataset unknown %s' % name)
  if name not in datasets_map:
    return default.get_split(
      split_name,
      dataset_dir,
      file_pattern,
      reader,
      train_image_count=train_image_count,
      val_image_count=val_image_count,
      label_count=label_count,
      data_name=name)

  return datasets_map[name].get_split(
      split_name,
      dataset_dir,
      file_pattern,
      reader,
      train_image_count=train_image_count,
      val_image_count=val_image_count,
      label_count=label_count)
