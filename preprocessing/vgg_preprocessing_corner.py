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

slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

#_RESIZE_SIDE_MIN = 256
#_RESIZE_SIDE_MAX = 512

## for image input size 400
_RESIZE_SIDE_MIN = 512
_RESIZE_SIDE_MAX = 768


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
  cropped_shape = control_flow_ops.with_dependencies(
      [rank_assertion],
      tf.stack([crop_height, crop_width, original_shape[2]]))

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  image = control_flow_ops.with_dependencies(
      [size_assertion],
      tf.slice(image, offsets, cropped_shape))
  return tf.reshape(image, cropped_shape)


def _random_crop_one(image, crop_height, crop_width):
  rank_assertions = []

  image_rank = tf.rank(image)
  rank_assert = tf.Assert(
      tf.equal(image_rank, 3),
      ['Wrong rank for tensor  %s [expected] [actual]',
       image.name, 3, image_rank])

  rank_assertions.append(rank_assert)
  image_shape = control_flow_ops.with_dependencies(
      [rank_assertions[0]],
      tf.shape(image))
  image_height = image_shape[0]
  image_width = image_shape[1]
  max_offset_height = control_flow_ops.with_dependencies(
      rank_assertions, tf.reshape(image_height - crop_height + 1, []))
  max_offset_width = control_flow_ops.with_dependencies(
      rank_assertions, tf.reshape(image_width - crop_width + 1, []))
  offset_height = tf.random_uniform(
      [], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform(
      [], maxval=max_offset_width, dtype=tf.int32)

  return _crop(image, offset_height, offset_width, crop_height, crop_width)


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

  image_shape = control_flow_ops.with_dependencies(
      [rank_assertions[0]],
      tf.shape(image_list[0]))
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
    shape = control_flow_ops.with_dependencies([rank_assertions[i]],
                                               tf.shape(image))
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
  max_offset_height = control_flow_ops.with_dependencies(
      asserts, tf.reshape(image_height - crop_height + 1, []))
  max_offset_width = control_flow_ops.with_dependencies(
      asserts, tf.reshape(image_width - crop_width + 1, []))
  offset_height = tf.random_uniform(
      [], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform(
      [], maxval=max_offset_width, dtype=tf.int32)

  return [_crop(image, offset_height, offset_width,
                crop_height, crop_width) for image in image_list]


def _corner_crop_random(image_list, crop_height, crop_width):
  """Performs for corner crops of the given image list.
    crop_height crop_width should be 224
    short edge of image is 512
    target is 224*224
    image divided into 16 patches
    for each patch short edge is 128, randorm crop into 112*112
    four corners paste into a 224*224 image
  """

  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    crop1 = tf.slice(image, [0, 0, 0], [tf.to_int32(image_height/4), tf.to_int32(image_width/4), 3])
    crop2 = tf.slice(image, [0, image_width-tf.to_int32(image_width/4), 0],
                     [tf.to_int32(image_height/4), tf.to_int32(image_width/4), 3])
    crop3 = tf.slice(image, [image_height-tf.to_int32(image_height/4), 0, 0],
                     [tf.to_int32(image_height/4), tf.to_int32(image_width/4), 3])
    crop4 = tf.slice(image, [image_height-tf.to_int32(image_height/4), image_width-tf.to_int32(image_width/4), 0],
                     [tf.to_int32(image_height/4), tf.to_int32(image_width/4), 3])

    crop1 = _random_crop_one(crop1, int(crop_height/2), int(crop_width/2))
    crop2 = _random_crop_one(crop2, int(crop_height/2), int(crop_width/2))
    crop3 = _random_crop_one(crop3, int(crop_height/2), int(crop_width/2))
    crop4 = _random_crop_one(crop4, int(crop_height/2), int(crop_width/2))

    result = tf.concat((tf.concat((crop1, crop2), axis=1), tf.concat((crop3, crop4), axis=1)), axis=0)

    outputs.append(result)
  return outputs


def _corner_crop(image_list, crop_height, crop_width):
  """Performs for corner crops of the given image list.
    crop_height crop_width should be 224
    short edge of image is 512
    target is 224*224
    image divided into 16 patches
    for each patch short edge is 128, randorm crop into 112*112
    four corners paste into a 224*224 image
  """

  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    crop1 = tf.slice(image, [0, 0, 0], [tf.to_int32(image_height/4), tf.to_int32(image_width/4), 3])
    crop2 = tf.slice(image, [0, image_width-tf.to_int32(image_width/4), 0],
                     [tf.to_int32(image_height/4), tf.to_int32(image_width/4), 3])
    crop3 = tf.slice(image, [image_height-tf.to_int32(image_height/4), 0, 0],
                     [tf.to_int32(image_height/4), tf.to_int32(image_width/4), 3])
    crop4 = tf.slice(image, [image_height-tf.to_int32(image_height/4), image_width-tf.to_int32(image_width/4), 0],
                     [tf.to_int32(image_height/4), tf.to_int32(image_width/4), 3])

    offset_height = (image_height/4 - crop_height/2) / 2
    offset_width = (image_width/4 - crop_width/2) / 2

    crop1 = _crop(crop1, offset_height, offset_width, tf.to_int32(crop_height/2), tf.to_int32(crop_width/2))
    crop2 = _crop(crop2, offset_height, offset_width, tf.to_int32(crop_height/2), tf.to_int32(crop_width/2))
    crop3 = _crop(crop3, offset_height, offset_width, tf.to_int32(crop_height/2), tf.to_int32(crop_width/2))
    crop4 = _crop(crop4, offset_height, offset_width, tf.to_int32(crop_height/2), tf.to_int32(crop_width/2))

    result = tf.concat((tf.concat((crop1, crop2), axis=1), tf.concat((crop3, crop4), axis=1)), axis=0)

    outputs.append(result)
  return outputs


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


def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)
  return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])
  return resized_image


def preprocess_for_train(image,
                         output_height,
                         output_width,
                         resize_side_min=_RESIZE_SIDE_MIN,
                         resize_side_max=_RESIZE_SIDE_MAX):
  """Preprocesses the given image for training.

  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """
  resize_side = tf.random_uniform(
      [], minval=resize_side_min, maxval=resize_side_max+1, dtype=tf.int32)

  image = _aspect_preserving_resize(image, resize_side)
  image = _corner_crop_random([image], output_height, output_width)[0]
  # image = _random_crop([image], output_height, output_width)[0]
  image.set_shape([output_height, output_width, 3])
  image = tf.to_float(image)
  image = tf.image.random_flip_left_right(image)
  return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def preprocess_for_eval(image, output_height, output_width, resize_side=_RESIZE_SIDE_MIN):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side: The smallest side of the image for aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """
  image = _aspect_preserving_resize(image, resize_side)
  image = _corner_crop([image], output_height, output_width)[0]
  # image = _central_crop([image], output_height, output_width)[0]
  image.set_shape([output_height, output_width, 3])

  # for test show crop result
  # return image

  image = tf.to_float(image)
  return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def preprocess_image(image, output_height, output_width, is_training=False,
                     resize_side_min=_RESIZE_SIDE_MIN,
                     resize_side_max=_RESIZE_SIDE_MAX, **kwargs):
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
  if is_training:
    return preprocess_for_train(image, output_height, output_width,
                                resize_side_min, resize_side_max)
  else:
    return preprocess_for_eval(image, output_height, output_width,
                               resize_side_min)



if __name__ == "__main__":
    import sys
    import numpy as np

    import matplotlib.pyplot as plt
    if len(sys.argv) < 2:
        print("Need image param")
        exit()
    img_file = sys.argv[1]

    image_data = tf.gfile.FastGFile(img_file, 'rb').read()
    input_tensor = tf.image.decode_jpeg(image_data)
    processed_tensor = preprocess_for_eval(input_tensor, 224, 224, _RESIZE_SIDE_MIN)
    sess = tf.Session()
    processed = sess.run(processed_tensor)
    # print("input_tensor: %s" % (input_tensor))
    # print("processed: %s" % (processed[112, :50, :]))

    with tf.Session() as sess:
        img_ = input_tensor.eval()
        print(img_.shape)
        # print("img_: %s" % (img_[int(img_.shape[0]/2), :50, :]))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(img_)
    plt.subplot(2, 1, 2)
    plt.imshow(np.array(processed).astype(np.int))
    plt.show()


