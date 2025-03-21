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
"""Provides utilities to preprocess images.

The preprocessing steps for VGG were introduced in the following technical
report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops

slim = tf.contrib.slim

_R_MEAN = 123.680
_G_MEAN = 116.779
_B_MEAN = 103.939
_R_STD = 58.393
_G_STD = 57.120
_B_STD = 57.375

_RESIZE_SIDE = 256


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]


def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.

  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.

  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

  Returns:
    the cropped (and resized) image.

  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.slice(image, offsets, cropped_shape)
  return tf.reshape(image, cropped_shape)


def _random_crop(image_list, crop_height, crop_width):
  """Crops the given list of images.

  The function applies the same crop to each image in the list. This can be
  effectively applied when there are multiple image inputs of the same
  dimension such as:

    image, depths, normals = _random_crop([image, depths, normals], 120, 150)

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the new height.
    crop_width: the new width.

  Returns:
    the image_list with cropped images.

  Raises:
    ValueError: if there are multiple image inputs provided with different size
      or the images are smaller than the crop dimensions.
  """
  if not image_list:
    raise ValueError('Empty image_list.')

  # Compute the rank assertions.
  rank_assertions = []
  for i in range(len(image_list)):
    image_rank = tf.rank(image_list[i])
    rank_assert = tf.Assert(
        tf.equal(image_rank, 3),
        ['Wrong rank for tensor  %s [expected] [actual]',
         image_list[i].name, 3, image_rank])
    rank_assertions.append(rank_assert)

  with tf.control_dependencies([rank_assertions[0]]):
    image_shape = tf.shape(image_list[0])
  image_height = image_shape[0]
  image_width = image_shape[1]
  crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(image_height, crop_height),
          tf.greater_equal(image_width, crop_width)),
      ['Crop size greater than the image size.'])

  asserts = [rank_assertions[0], crop_size_assert]

  for i in range(1, len(image_list)):
    image = image_list[i]
    asserts.append(rank_assertions[i])
    with tf.control_dependencies([rank_assertions[i]]):
      shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    height_assert = tf.Assert(
        tf.equal(height, image_height),
        ['Wrong height for tensor %s [expected][actual]',
         image.name, height, image_height])
    width_assert = tf.Assert(
        tf.equal(width, image_width),
        ['Wrong width for tensor %s [expected][actual]',
         image.name, width, image_width])
    asserts.extend([height_assert, width_assert])

  # Create a random bounding box.
  #
  # Use tf.random_uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  with tf.control_dependencies(asserts):
    max_offset_height = tf.reshape(image_height - crop_height + 1, [])
  with tf.control_dependencies(asserts):
    max_offset_width = tf.reshape(image_width - crop_width + 1, [])
  offset_height = tf.random_uniform(
      [], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform(
      [], maxval=max_offset_width, dtype=tf.int32)

  return [_crop(image, offset_height, offset_width,
                crop_height, crop_width) for image in image_list]


def _central_crop(image_list, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    outputs.append(_crop(image, offset_height, offset_width,
                         crop_height, crop_width))
  return outputs


def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)


def _image_normalization(image, means, stds):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    stds = [58.393, 57.12, 57.375]
    image = _image_normalization(image, means, stds)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.
    stds: a C-vector of values to division from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] = (channels[i] - means[i]) / stds[i]
  return tf.concat(axis=2, values=channels)


def _aspect_changing_resize(image, resize_side):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    resize_side: A python integer or scalar `Tensor` indicating the size of
      the side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(image, [resize_side, resize_side],
                                           align_corners=False)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])
  return resized_image


def preprocess_for_train(image, height, width, resize_side):
  """Preprocesses the given image for training.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    height: The height of the image after preprocessing.
    width: The width of the image after preprocessing.
    resize_side: The size of the side after resizing.

  Returns:
    A preprocessed image.
  """
  tf.logging.info("******* scene_preprocessing preprocess_for_train resize_side: %s", resize_side)
  image = _aspect_changing_resize(image, resize_side)
  image = _random_crop([image], height, width)[0]
  image.set_shape([height, width, 3])
  image = tf.to_float(image)
  image = tf.image.random_flip_left_right(image)
  return _image_normalization(image, [_R_MEAN, _G_MEAN, _B_MEAN],
                              [_R_STD, _G_STD, _B_STD])


def preprocess_for_eval(image, height, width, resize_side):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side: The smallest side of the image for aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """
  #image = _aspect_changing_resize(image, resize_side)
  #image = _central_crop([image], height, width)[0]
  image = _aspect_changing_resize(image, height)
  image.set_shape([height, width, 3])
  image = tf.to_float(image)
  return _image_normalization(image, [_R_MEAN, _G_MEAN, _B_MEAN],
                              [_R_STD, _G_STD, _B_STD])


def preprocess_image(image, height, width, is_training=False,
                     resize_side=_RESIZE_SIDE, **kwargs):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, then this value
      is used for rescaling.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, this value is
      ignored. Otherwise, the resize side is sampled from
        [resize_size_min, resize_size_max].

  Returns:
    A preprocessed image.
  """

  # adapt for different target size
  resize_side = int(height * 1.1428)
  if is_training:
    return preprocess_for_train(image, height, width, resize_side)
  else:
    return preprocess_for_eval(image, height, width, resize_side)

# preprocessing method for automl group
class scene_processing_automl:
    @staticmethod
    def preprocess_image(image, height, width, is_training=False,
                         resize_side=_RESIZE_SIDE, **kwargs):
      resize_side = int(height * 1.1428)
      if is_training:
        return preprocess_for_train.preprocess_for_train_new(image, height, width, resize_side)
      else:
        return scene_processing_automl.preprocess_for_eval_automl(image, height, width, resize_side)

    @staticmethod
    def preprocess_for_eval_automl(image, height, width, resize_side):
      image = _aspect_changing_resize(image, height)
      image.set_shape([height, width, 3])
      image = tf.to_float(image)
      _R_MEAN_automl = 123.675
      _G_MEAN_automl = 116.28
      _B_MEAN_automl = 103.53
      _R_STD_automl = 64.0
      _G_STD_automl = 64.0
      _B_STD_automl = 64.0
      return _image_normalization(image, [_R_MEAN_automl, _G_MEAN_automl, _B_MEAN_automl],
                                  [_R_STD_automl, _G_STD_automl, _B_STD_automl])


class scene_new_processing:
    @staticmethod
    def preprocess_image(image, height, width, is_training=False,
                         resize_side=_RESIZE_SIDE, **kwargs):
      resize_side = int(height * 1.1428)
      if is_training:
        return scene_new_processing.preprocess_for_train_new(image, height, width, resize_side)
      else:
        return preprocess_for_eval(image, height, width, resize_side)

    @staticmethod
    def preprocess_for_train_new(image, height, width, resize_side):

        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                           dtype=tf.float32,
                           shape=[1, 1, 4])

        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=0.81,
            aspect_ratio_range=(0.5, 2),
            area_range=(0.81, 1.0),
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        image = tf.slice(image, bbox_begin, bbox_size)
        image.set_shape([None, None, 3])

        image = _aspect_changing_resize(image, height)
        # We select only 1 case for fast_mode bilinear.
        '''
        fast_mode = False
        num_resize_cases = 1 if fast_mode else 4
        image = apply_with_random_selector(
            image,
            lambda x, method: tf.image.resize_images(x, [height, width], method=method),
            num_cases=num_resize_cases)
        '''

        image = tf.to_float(image)
        image = tf.image.random_flip_left_right(image)

        return _image_normalization(image, [_R_MEAN, _G_MEAN, _B_MEAN],
                                    [_R_STD, _G_STD, _B_STD])

class scene_new_mtk_processing:
    @staticmethod
    def preprocess_image(image, height, width, is_training=False,
                         resize_side=_RESIZE_SIDE, **kwargs):
      resize_side = int(height * 1.1428)
      if is_training:
        return scene_new_processing.preprocess_for_train_new(image, height, width, resize_side)
      else:
        return preprocess_for_eval(image, height, width, resize_side)

    @staticmethod
    def preprocess_for_train_new(image, height, width, resize_side):

        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                           dtype=tf.float32,
                           shape=[1, 1, 4])

        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=0.81,
            aspect_ratio_range=(0.5, 2),
            area_range=(0.81, 1.0),
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        image = tf.slice(image, bbox_begin, bbox_size)
        image.set_shape([None, None, 3])

        image = _aspect_changing_resize(image, height)
        # We select only 1 case for fast_mode bilinear.
        '''
        fast_mode = False
        num_resize_cases = 1 if fast_mode else 4
        image = apply_with_random_selector(
            image,
            lambda x, method: tf.image.resize_images(x, [height, width], method=method),
            num_cases=num_resize_cases)
        '''

        image = tf.to_float(image)
        image = tf.image.random_flip_left_right(image)

        #随机左右旋转
        uniform_random = random_ops.random_uniform([], 0, 3.0)
        k = tf.to_int32(math_ops.floor(uniform_random), 'rot_k')
        if(k==2):
            k=k+1
        image = tf.image._rot90(image, k=k)

        return _image_normalization(image, [_R_MEAN, _G_MEAN, _B_MEAN],
                                    [_R_STD, _G_STD, _B_STD])




class scene_new_2_processing:
    @staticmethod
    def preprocess_image(image, height, width, is_training=False,
                         resize_side=_RESIZE_SIDE, **kwargs):
      resize_side = int(height * 1.1428)
      if is_training:
        return scene_new_processing.preprocess_for_train_new(image, height, width, resize_side)
      else:
        return preprocess_for_eval(image, height, width, resize_side)

    @staticmethod
    def preprocess_for_train_new(image, height, width, resize_side):

        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                           dtype=tf.float32,
                           shape=[1, 1, 4])

        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=0.81,
            aspect_ratio_range=(0.5, 2),
            area_range=(0.81, 1.0),
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        image = tf.slice(image, bbox_begin, bbox_size)
        image.set_shape([None, None, 3])

        image = tf.to_float(image)
        # We select only 1 case for fast_mode bilinear.
        fast_mode = False
        num_resize_cases = 1 if fast_mode else 4
        image = apply_with_random_selector(
            image,
            lambda x, method: tf.image.resize_images(x, [height, width], method=method),
            num_cases=num_resize_cases)

        image = tf.image.random_flip_left_right(image)
        return _image_normalization(image, [_R_MEAN, _G_MEAN, _B_MEAN],
                                    [_R_STD, _G_STD, _B_STD])


class scene_new_processing_distill:
    @staticmethod
    def preprocess_image(image, image_size_L, image_size_S, is_training=False, **kwargs):
      if is_training:
        return scene_new_processing_distill.preprocess_for_train_distill(image, image_size_L, image_size_S)
      else:
        return preprocess_for_eval(image, image_size_L, image_size_L)

    @staticmethod
    def preprocess_for_train_distill(image, image_size_L, image_size_S):

        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                           dtype=tf.float32,
                           shape=[1, 1, 4])

        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=0.81,
            aspect_ratio_range=(0.5, 2),
            area_range=(0.81, 1.0),
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        image = tf.slice(image, bbox_begin, bbox_size)
        image.set_shape([None, None, 3])

        imageL = _aspect_changing_resize(image, image_size_L)
        # We select only 1 case for fast_mode bilinear.
        '''
        fast_mode = False
        num_resize_cases = 1 if fast_mode else 4
        image = apply_with_random_selector(
            image,
            lambda x, method: tf.image.resize_images(x, [height, width], method=method),
            num_cases=num_resize_cases)
        '''

        imageL = tf.to_float(imageL)
        imageL = tf.image.random_flip_left_right(imageL)
        imageS = _aspect_changing_resize(imageL, image_size_S)
        return _image_normalization(imageL, [_R_MEAN, _G_MEAN, _B_MEAN], [_R_STD, _G_STD, _B_STD]), \
                _image_normalization(imageS, [_R_MEAN, _G_MEAN, _B_MEAN], [_R_STD, _G_STD, _B_STD])


class scene_new_no_norm_processing:
    @staticmethod
    def preprocess_image(image, height, width, is_training=False,
                         resize_side=_RESIZE_SIDE, **kwargs):
      if is_training:
        return scene_new_no_norm_processing.preprocess_for_train(image, height, width, resize_side)
      else:
        return scene_new_no_norm_processing.preprocess_for_eval(image, height, width, resize_side)

    @staticmethod
    def preprocess_for_train(image, height, width, resize_side):

        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                           dtype=tf.float32,
                           shape=[1, 1, 4])

        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=0.81,
            aspect_ratio_range=(0.5, 2),
            area_range=(0.81, 1.0),
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        image = tf.slice(image, bbox_begin, bbox_size)
        image.set_shape([None, None, 3])

        image = _aspect_changing_resize(image, height)
        # We select only 1 case for fast_mode bilinear.
        '''
        fast_mode = False
        num_resize_cases = 1 if fast_mode else 4
        image = apply_with_random_selector(
            image,
            lambda x, method: tf.image.resize_images(x, [height, width], method=method),
            num_cases=num_resize_cases)
        '''

        image = tf.to_float(image)
        image = tf.image.random_flip_left_right(image)
        return image

    @staticmethod
    def preprocess_for_eval(image, height, width, resize_side):
        """Preprocesses the given image for evaluation.

        Args:
          image: A `Tensor` representing an image of arbitrary size.
          output_height: The height of the image after preprocessing.
          output_width: The width of the image after preprocessing.
          resize_side: The smallest side of the image for aspect-preserving resizing.

        Returns:
          A preprocessed image.
        """
        # image = _aspect_changing_resize(image, resize_side)
        # image = _central_crop([image], height, width)[0]
        image = _aspect_changing_resize(image, height)
        image.set_shape([height, width, 3])
        image = tf.to_float(image)
        return image