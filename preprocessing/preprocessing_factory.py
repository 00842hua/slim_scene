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
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from preprocessing import cifarnet_preprocessing
from preprocessing import inception_preprocessing
from preprocessing import lenet_preprocessing
from preprocessing import vgg_preprocessing
from preprocessing import vgg_preprocessing_corner
from preprocessing import inception_mod_preprocessing
from preprocessing import vgg_mod_preprocessing
from preprocessing import scene_preprocessing


slim = tf.contrib.slim


def get_preprocessing(name, is_training=False):
  """Returns preprocessing_fn(image, height, width, **kwargs).

  Args:
    name: The name of the preprocessing function.
    is_training: `True` if the model is being used for training and `False`
      otherwise.

  Returns:
    preprocessing_fn: A function that preprocessing a single image (pre-batch).
      It has the following signature:
        image = preprocessing_fn(image, output_height, output_width, ...).

  Raises:
    ValueError: If Preprocessing `name` is not recognized.
  """
  preprocessing_fn_map = {
      'cifarnet': cifarnet_preprocessing,
      'inception': inception_preprocessing,
      'inception_v1': inception_preprocessing,
      'inception_v2': inception_preprocessing,
      'inception_v3': inception_preprocessing,
      'inception_v4': inception_preprocessing,
      'inception_resnet_v2': inception_preprocessing,
      'inception_resnet_v2_center': inception_preprocessing,
      'inception_resnet_v2_dropblock': inception_preprocessing,
      'lenet': lenet_preprocessing,
      'mobilenet_v1': inception_preprocessing,
      'mobilenet_v2': inception_preprocessing,
      'mobilenet_v2_035': inception_preprocessing,
      'mobilenet_v2_140': inception_preprocessing,
      'nasnet_mobile': inception_preprocessing,
      'nasnet_large': inception_preprocessing,
      'pnasnet_mobile': inception_preprocessing,
      'pnasnet_large': inception_preprocessing,
      'inception_vpn': inception_preprocessing.vpn_processing,
      'inception_vpn_notcrop': inception_preprocessing.vpn_notcrop_processing,
      'inception_notcrop': inception_preprocessing.notcrop_processing,
      'inception_crop_little': inception_preprocessing.crop_little_processing,
      'resnet_v1_50': vgg_preprocessing,
      'resnet_v1_101': vgg_preprocessing,
      'resnet_v1_152': vgg_preprocessing,
      'resnet_v1_200': vgg_preprocessing,
      'resnet_v2_50': vgg_preprocessing,
      'resnet_v2_101': vgg_preprocessing,
      'resnet_v2_152': vgg_preprocessing,
      'resnet_v2_200': vgg_preprocessing,
      'vgg': vgg_preprocessing,
      'vgg_a': vgg_preprocessing,
      'vgg_16': vgg_preprocessing,
      'vgg_19': vgg_preprocessing,
      'vgg_scene_new': vgg_preprocessing.vgg_scene_new_processing,
      'vgg_inception': vgg_preprocessing.vgg_inception_processing,
      'vgg_corner': vgg_preprocessing_corner,
      'inception_all': inception_mod_preprocessing.inception_preprocess_all,
      'inception_common': inception_mod_preprocessing.inception_preprocess_common,
      'inception_common_ori': inception_mod_preprocessing.inception_preprocess_common_ori,
      'vgg_all': vgg_mod_preprocessing.inception_preprocess_all,
      'vgg_all_400': vgg_mod_preprocessing.inception_preprocess_all_400,
      'inception_common_min_crop': inception_mod_preprocessing.inception_preprocess_common_min_crop,
      'inception_preprocess_common_min_crop_mul_size': inception_mod_preprocessing.inception_preprocess_common_min_crop_mul_size,
      'scene': scene_preprocessing,
      'scene_new': scene_preprocessing.scene_new_processing,
      'scene_new_distill': scene_preprocessing.scene_new_processing_distill,
      'scene_new_2': scene_preprocessing.scene_new_2_processing,
      'scene_new_mtk': scene_preprocessing.scene_new_mtk_processing,
      'scene_automl': scene_preprocessing.scene_processing_automl,
      'scene_new_no_norm': scene_preprocessing.scene_new_no_norm_processing,
  }

  if name not in preprocessing_fn_map:
    raise ValueError('Preprocessing name [%s] was not recognized' % name)

  def preprocessing_fn(image, output_height, output_width, **kwargs):
    return preprocessing_fn_map[name].preprocess_image(
        image, output_height, output_width, is_training=is_training, **kwargs)

  return preprocessing_fn
