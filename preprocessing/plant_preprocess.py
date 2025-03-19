import random

import time

from PIL import Image

from augmentor import PipelineFly
import tensorflow as tf
import numpy as np
from preprocessing.inception_preprocessing_resample import apply_with_random_selector, distort_color

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512


def inception_preprocess_fn_eval(file, img_size):
    img = Image.open(file)
    img_w, img_h = img.size
    shortsize = img_w if img_h > img_w else img_h

    if shortsize == img_w:
        longsize = min(img_h, int(shortsize * 1.5))
        crop_w = shortsize
        crop_h = longsize
    else:
        longsize = min(img_w, int(shortsize * 1.5))
        crop_w = longsize
        crop_h = shortsize

    print("crop_w, crop_h:{}, {}".format(crop_w, crop_h))
    pipeline_fly = PipelineFly()
    pipeline_fly.crop_by_size(1, crop_w, crop_h, True)
    pipeline_fly.resize(1, img_size, img_size, "BILINEAR")
    dist_img = pipeline_fly(img)
    np_dist_img = np.asarray(dist_img)
    np_dist_img = np_dist_img[..., :3]
    # debug = True
    # if debug:
    #     dist_img.save("/ceph_plant/xinpeng/tmp/test.JPEG")
    return np_dist_img


def _aspect_preserving_resize(image, smallest_side):
    img_w, img_h = image.size

    scale = smallest_side * 1.0 / img_w if img_h > img_w else smallest_side * 1.0 / img_h

    new_height = int(round(img_h * scale))
    new_width = int(round(img_w * scale))
    return image.resize((new_width, new_height), Image.BICUBIC)


def _central_crop(image, crop_width, crop_height):
    img_w, img_h = image.size

    offset_height = int((img_h - crop_height) / 2)
    offset_width = int((img_w - crop_width) / 2)

    return image.crop((offset_width, offset_height, crop_width + offset_width,
                       crop_height + offset_height))


def vgg_preprocess_fn_train(file, img_size):
    resize_side_min = _RESIZE_SIDE_MIN
    resize_side_max = _RESIZE_SIDE_MAX
    resize_side = random.randint(resize_side_min, resize_side_max + 1)
    img = Image.open(file)
    image = _aspect_preserving_resize(img, resize_side)
    pipeline_fly = PipelineFly()
    # trans_prob = random.random()
    # if trans_prob < 0.5:
    #     pipeline_fly.rotate(1, 12, 12)
    # elif trans_prob < 0.8:
    #     pipeline_fly.skew(1, 0.8)
    # elif trans_prob < 0.9:
    #     pipeline_fly.shear(1, 8, 8)
    #
    # distort_prob = random.random()
    # if distort_prob < 0.1:
    #     pipeline_fly.gaussian_distortion(1, 5, 5, random.randint(2, 8), "bell", "in")
    # elif distort_prob < 0.2:
    #     pipeline_fly.random_distortion(1, 5, 5, random.randint(2, 8))

    pipeline_fly.crop_by_size(1.0, img_size, img_size, False)
    dist_img = pipeline_fly(image)
    np_dist_img = np.asarray(dist_img)
    np_dist_img = np_dist_img[..., :3]
    return np_dist_img


def vgg_preprocess_fn_eval(file, img_size):
    img = Image.open(file)
    raw_w, raw_h = img.size
    smallest_side = min(min(raw_w, raw_h), img_size)
    # resize_side_min = _RESIZE_SIDE_MIN
    # resize_side = resize_side_min if img_size < resize_side_min else (img_size + 10)

    image = _aspect_preserving_resize(img, smallest_side + 32)

    img_w, img_h = image.size
    img_ratio = img_w * 1.0 / img_h
    max_ratio = 1.5 #(16.0 / 9.0)
    if img_ratio > max_ratio:
        crop_w = int(max_ratio * smallest_side)
        crop_h = smallest_side
    elif img_ratio < (1.0 / max_ratio):
        crop_w = smallest_side
        crop_h = int(max_ratio * smallest_side)
    elif img_ratio > 1:
        crop_w = int(img_ratio * smallest_side)
        crop_h = smallest_side
    else:
        crop_w = smallest_side
        crop_h = int(smallest_side / img_ratio)

    image = _central_crop(image, crop_w, crop_h)
    np_dist_img = np.asarray(image)
    np_dist_img = np_dist_img[..., :3]
    debug = False
    if debug:
        now = time.time()
        img.save("/ceph_plant/xinpeng/tmp/{}_raw.JPEG".format(now))
        image.save("/ceph_plant/xinpeng/tmp/{}_test.JPEG".format(now))
    return np_dist_img


def inception_preprocess_fn_train(file, img_size):
    img = Image.open(file)
    pipeline_fly = PipelineFly()

    pipeline_fly.crop_distort(1)
    rize_type = ["BICUBIC", "BILINEAR", "ANTIALIAS", "NEAREST"]
    pipeline_fly.resize(1, img_size, img_size, random.choice(rize_type))
    # pipeline_fly.flip_left_right(0.5)

    # pipeline_fly.rotate(1, 12, 12)
    # trans_prob = random.random()
    # if trans_prob < 0.5:
    #     pipeline_fly.skew(1, 0.8)
    # elif trans_prob < 0.7:
    #     pipeline_fly.shear(1, 8, 8)
    #
    # distort_prob = random.random()
    # if distort_prob < 0.1:
    #     pipeline_fly.gaussian_distortion(1, 5, 5, random.randint(2, 8), "bell", "in")
    # elif distort_prob < 0.2:
    #     pipeline_fly.random_distortion(1, 5, 5, random.randint(2, 8))
    #
    # pipeline_fly.random_erasing(0.5, 0.4)

    dist_img = pipeline_fly(img)
    np_dist_img = np.asarray(dist_img)
    np_dist_img = np_dist_img[..., :3]
    is_debug = False
    if is_debug:
        id = random.randint(0, 100000)
        Image.open(file).save("/data/xinpeng/tmp_pic/{}_raw.JPEG".format(id))
        dist_img.save("/data/xinpeng/tmp_pic/{}_pip.JPEG".format(id))
    return np_dist_img


def inception_preprocess(file, img_size, is_train=True):
    if is_train:
        return inception_preprocess_fn_train(file, img_size)
    else:
        return inception_preprocess_fn_eval(file, img_size)


def vgg_preprocess(file, img_size, is_train=True):
    if is_train:
        return vgg_preprocess_fn_train(file, img_size)
    else:
        return vgg_preprocess_fn_eval(file, img_size)


preprocessing_fn_map = {
      'inception': inception_preprocess,
      'inception_v1': inception_preprocess,
      'inception_v2': inception_preprocess,
      'inception_v3': inception_preprocess,
      'inception_v4': inception_preprocess,
      'inception_resnet_v2': inception_preprocess,
      'inception_resnet_v2_se': inception_preprocess,
      'resnet_v1_50': vgg_preprocess,
      'resnet_v1_101': vgg_preprocess,
      'resnet_v1_152': vgg_preprocess,
      'resnet_v1_200': vgg_preprocess,
      'resnet_v2_50': vgg_preprocess,
      'resnet_v2_101': vgg_preprocess,
      'resnet_v2_152': vgg_preprocess,
      'resnet_v2_200': vgg_preprocess,
  }


def preprocess_fn(name):
    if name not in preprocessing_fn_map:
        raise ValueError('Prerocessing name [%s] was not recognized' % name)
    tf.logging.info("get_preprocessing:{}".format(name))
    return preprocessing_fn_map[name]


def inception_postprocess(image, is_train=True):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if is_train:
        image = random_flip_left_right(image)
        image = apply_with_random_selector(
            image,
            lambda x, ordering: distort_color(x, ordering, False),
            num_cases=4)

    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


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
    if image.get_shape().ndims != 4:
        raise ValueError('Input must be of size [batch, height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def random_flip_left_right(image, seed=None):
  """Randomly flip an image horizontally (left to right).

  With a 1 in 2 chance, outputs the contents of `image` flipped along the
  second dimension, which is `width`.  Otherwise output the image as-is.

  Args:
    image: A 4-D tensor of shape `[batch, height, width, channels].`
    seed: A Python integer. Used to create a random seed. See
      @{tf.set_random_seed}
      for behavior.

  Returns:
    A 4-D tensor of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  """
  with tf.name_scope(None, 'random_flip_left_right', [image]) as scope:
    image = tf.convert_to_tensor(image, name='image')

    uniform_random = tf.random_uniform([], 0, 1.0, seed=seed)
    mirror_cond = tf.less(uniform_random, .5)
    result = tf.cond(
        mirror_cond,
        lambda: tf.reverse(image, [2]),
        lambda: image,
        name=scope)
    result.set_shape(image.get_shape())
    return result


def vgg_postprocess(image, is_train=True):
    if is_train:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = random_flip_left_right(image)
        # image = apply_with_random_selector(
        #     image,
        #     lambda x, ordering: distort_color(x, ordering, False),
        #     num_cases=4)
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    image = tf.to_float(image)
    return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


postprocessing_fn_map = {
      'inception': inception_postprocess,
      'inception_v1': inception_postprocess,
      'inception_v2': inception_postprocess,
      'inception_v3': inception_postprocess,
      'inception_v4': inception_postprocess,
      'inception_resnet_v2': inception_postprocess,
      'inception_resnet_v2_se': inception_postprocess,
      'resnet_v1_50': vgg_postprocess,
      'resnet_v1_101': vgg_postprocess,
      'resnet_v1_152': vgg_postprocess,
      'resnet_v1_200': vgg_postprocess,
      'resnet_v2_50': vgg_postprocess,
      'resnet_v2_101': vgg_postprocess,
      'resnet_v2_152': vgg_postprocess,
      'resnet_v2_200': vgg_postprocess,
  }


def postprocess_fn(name, is_train=True):
    if name not in postprocessing_fn_map:
        raise ValueError('Postrocessing name [%s] was not recognized' % name)
    tf.logging.info("get_preprocessing:{}".format(name))

    def postprocessing_fn(image, add_image_summaries=False):
        if add_image_summaries:
            with tf.device("/device:CPU:0"):
                pre_img = image
                tf.summary.image('image_preprocess', pre_img, max_outputs=4)
        image = postprocessing_fn_map[name](image, is_train=is_train)
        if add_image_summaries:
            with tf.device("/device:CPU:0"):
                post_img = image
                tf.summary.image('image_postprocess', post_img, max_outputs=4)
        return image

    return postprocessing_fn
