#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import check_ops

from preprocessing import inception_preprocessing

import math
import sys
sys.path.append("..")
import tf_version as tv


def _Check3DImage(image, require_static=True):
    """Assert that we are working with properly shaped image.

    Args:
    image: 3-D Tensor of shape [height, width, channels]
    require_static: If `True`, requires that all dimensions of `image` are
      known and non-zero.

    Raises:
    ValueError: if `image.shape` is not a 3-vector.

    Returns:
    An empty list, if `image` has fully defined dimensions. Otherwise, a list
    containing an assert op is returned.
    """
    try:
        image_shape = image.get_shape().with_rank(3)
    except ValueError:
        raise ValueError("'image' must be three-dimensional.")
    if require_static and not image_shape.is_fully_defined():
        raise ValueError("'image' must be fully defined.")
    if any(x == 0 for x in image_shape):
        raise ValueError("all dims of 'image.shape' must be > 0: %s" %
                         image_shape)
    if not image_shape.is_fully_defined():
        return [check_ops.assert_positive(array_ops.shape(image),
                                          ["all dims of 'image.shape' "
                                           "must be > 0."])]
    else:
        return []


def _rot90(image, k=1, name=None):
    """Rotate an image counter-clockwise by 90 degrees.

    Args:
    image: A 3-D tensor of shape `[height, width, channels]`.
    k: A scalar integer. The number of times the image is rotated by 90 degrees.
    name: A name for this operation (optional).

    Returns:
    A rotated 3-D tensor of the same type and shape as `image`.
    """
    with ops.name_scope(name, 'rot90', [image, k]) as scope:
        image = ops.convert_to_tensor(image, name='image')
        _Check3DImage(image, require_static=False)
        k = ops.convert_to_tensor(k, dtype=dtypes.int32, name='k')
        k.get_shape().assert_has_rank(0)
        k = math_ops.mod(k, 4)

        def _rot90():
            return array_ops.transpose(array_ops.reverse(image, [False, True, False]),
                                       [1, 0, 2])

        def _rot180():
            return array_ops.reverse(image, [True, True, False])

        def _rot270():
            return array_ops.reverse(array_ops.transpose(image, [1, 0, 2]),
                                     [False, True, False])
        cases = [(math_ops.equal(k, 1), _rot90),
                 (math_ops.equal(k, 2), _rot180),
                 (math_ops.equal(k, 3), _rot270)]

        ret = control_flow_ops.case(cases, default=lambda: image, exclusive=True,
                                    name=scope)
        ret.set_shape(image.get_shape())
        return ret


def random_rot_image(image, seed=None, name=None):
    with ops.name_scope(name, 'random_rot', [image, seed]) as scope:
        uniform_random = random_ops.random_uniform([], 0, 2.0, seed=seed)
        k = tf.to_int32(math_ops.floor(uniform_random), 'rot_k')
        image = _rot90(image, k=k)

        return image


def preprocess_for_train(image, height, width, bbox,
                         distort_color,
                         crop_area_range=(0.05, 1.0),
                         aspect_ratio_range=(0.75, 1.33),
                         fast_mode=True,
                         rot_mode=False,
                         scope=None):
    with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
        if bbox is None:
            bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                               dtype=tf.float32,
                               shape=[1, 1, 4])
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                      bbox)
        '''
        if scope == None:
            tv.image_summary('image_with_bounding_boxes', image_with_box)
        else:
            tv.image_summary('image_with_bounding_boxes_'+scope, image_with_box)
        '''

        distorted_image, distorted_bbox = \
            inception_preprocessing.distorted_bounding_box_crop(image, bbox,
                                                                area_range=crop_area_range,
                                                                aspect_ratio_range=aspect_ratio_range)
        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([None, None, 3])
        image_with_distorted_box = tf.image.draw_bounding_boxes(
            tf.expand_dims(image, 0), distorted_bbox)
        '''
        if scope == None:
            tv.image_summary('images_with_distorted_bounding_box',
                             image_with_distorted_box)
        else:
            tv.image_summary('images_with_distorted_bounding_box_'+scope,
                             image_with_distorted_box)
        '''

        # This resizing operation may distort the images because the aspect
        # ratio is not respected. We select a resize method in a round robin
        # fashion based on the thread number.
        # Note that ResizeMethod contains 4 enumerated resizing methods.

        # We select only 1 case for fast_mode bilinear.
        num_resize_cases = 1 if fast_mode else 4
        distorted_image = inception_preprocessing.apply_with_random_selector(
            distorted_image,
            lambda x, method: tf.image.resize_images(x, [height, width], method=method),
            num_cases=num_resize_cases)
        '''
        if scope == None:
            tv.image_summary('cropped_resized_image',
                             tf.expand_dims(distorted_image, 0))
        else:
            tv.image_summary('cropped_resized_image_'+scope,
                             tf.expand_dims(distorted_image, 0))
        '''
        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        if rot_mode:
            distorted_image = random_rot_image(distorted_image)

        # Randomly distort the colors. There are 4 ways to do it.
        if distort_color:
            distorted_image = inception_preprocessing.apply_with_random_selector(
                distorted_image,
                lambda x, ordering: inception_preprocessing.distort_color(x, ordering, fast_mode),
                num_cases=4)
        '''
        if scope==None:
            tv.image_summary('final_distorted_image',
                             tf.expand_dims(distorted_image, 0))
        else:
            tv.image_summary('final_distorted_image_'+scope,
                             tf.expand_dims(distorted_image, 0))
        '''
        distorted_image = tv.sub(distorted_image, 0.5)
        distorted_image = tv.mul(distorted_image, 2.0)

        return distorted_image


def preprocess_for_train_mul_size(image, size_L, size_S, bbox,
                         distort_color,
                         crop_area_range=(0.05, 1.0),
                         aspect_ratio_range=(0.75, 1.33),
                         fast_mode=True,
                         rot_mode=False,
                         scope=None):
    with tf.name_scope(scope, 'distort_image', [image, size_L, size_S, bbox]):
        if bbox is None:
            bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                               dtype=tf.float32,
                               shape=[1, 1, 4])
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                      bbox)
        '''
        if scope == None:
            tv.image_summary('image_with_bounding_boxes', image_with_box)
        else:
            tv.image_summary('image_with_bounding_boxes_'+scope, image_with_box)
        '''

        distorted_image, distorted_bbox = \
            inception_preprocessing.distorted_bounding_box_crop(image, bbox,
                                                                area_range=crop_area_range,
                                                                aspect_ratio_range=aspect_ratio_range)
        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([None, None, 3])
        image_with_distorted_box = tf.image.draw_bounding_boxes(
            tf.expand_dims(image, 0), distorted_bbox)
        '''
        if scope == None:
            tv.image_summary('images_with_distorted_bounding_box',
                             image_with_distorted_box)
        else:
            tv.image_summary('images_with_distorted_bounding_box_'+scope,
                             image_with_distorted_box)
        '''

        # This resizing operation may distort the images because the aspect
        # ratio is not respected. We select a resize method in a round robin
        # fashion based on the thread number.
        # Note that ResizeMethod contains 4 enumerated resizing methods.

        # We select only 1 case for fast_mode bilinear.
        num_resize_cases = 1 if fast_mode else 4
        distorted_image = inception_preprocessing.apply_with_random_selector(
            distorted_image,
            lambda x, method: tf.image.resize_images(x, [size_L, size_L], method=method),
            num_cases=num_resize_cases)
        '''
        if scope == None:
            tv.image_summary('cropped_resized_image',
                             tf.expand_dims(distorted_image, 0))
        else:
            tv.image_summary('cropped_resized_image_'+scope,
                             tf.expand_dims(distorted_image, 0))
        '''
        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        if rot_mode:
            distorted_image = random_rot_image(distorted_image)

        # Randomly distort the colors. There are 4 ways to do it.
        if distort_color:
            distorted_image = inception_preprocessing.apply_with_random_selector(
                distorted_image,
                lambda x, ordering: inception_preprocessing.distort_color(x, ordering, fast_mode),
                num_cases=4)
        '''
        if scope==None:
            tv.image_summary('final_distorted_image',
                             tf.expand_dims(distorted_image, 0))
        else:
            tv.image_summary('final_distorted_image_'+scope,
                             tf.expand_dims(distorted_image, 0))
        '''
        distorted_image_2 = tf.image.resize_images(distorted_image, [size_S, size_S])

        distorted_image = tv.sub(distorted_image, 0.5)
        distorted_image = tv.mul(distorted_image, 2.0)

        distorted_image_2 = tv.sub(distorted_image_2, 0.5)
        distorted_image_2 = tv.mul(distorted_image_2, 2.0)

        return distorted_image, distorted_image_2

def preprocess_for_eval(image, height, width,
                        resize_method=tf.image.ResizeMethod.BILINEAR,
                        central_fraction=0.875, scope=None):
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        if central_fraction:
            image = tf.image.central_crop(image, central_fraction=central_fraction)

        if height and width:
            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_images(image, [height, width],
                                           method=resize_method,
                                           align_corners=False)
            image = tf.squeeze(image, [0])

        image = tv.sub(image, 0.5)
        image = tv.mul(image, 2.0)

        return image


def preprocess_image_base(image, height, width,
                          is_training=False,
                          bbox=None,
                          fast_mode=True,
                          scope=None,
                          distort_color=True,
                          rot_mode=False,
                          crop_area_range=(0.05, 1.0)):
    if is_training:
        return preprocess_for_train(image, height, width, bbox, distort_color,
                                    crop_area_range=crop_area_range,
                                    rot_mode=rot_mode,
                                    fast_mode=fast_mode, scope=scope)
    else:
        return inception_preprocessing.preprocess_for_eval(image, height, width)


class inception_preprocess_crop_20:
    @staticmethod
    def preprocess_image(image, height, width,
                     is_training=False,
                     bbox=None,
                     fast_mode=True,
                     scope=None):
        return preprocess_image_base(image, height, width,
                                     is_training=is_training,
                                     bbox=bbox,
                                     fast_mode=fast_mode,
                                     scope=scope,
                                     distort_color=True,
                                     crop_area_range=[0.2, 1.0])


class inception_preprocess_no_discolor:
    @staticmethod
    def preprocess_image(image, height, width,
                     is_training=False,
                     bbox=None,
                     fast_mode=True,
                     scope=None):
        return preprocess_image_base(image, height, width,
                                     is_training=is_training,
                                     bbox=bbox,
                                     fast_mode=fast_mode,
                                     scope=scope,
                                     distort_color=False,
                                     crop_area_range=[0.05, 1.0])


class inception_preprocess_no_crop:
    @staticmethod
    def preprocess_image(image, height, width,
                         is_training=False,
                         bbox=None,
                         fast_mode=True,
                         scope=None):
        if is_training:
            return inception_preprocessing.preprocess_for_train(image, height, width, bbox, fast_mode, scope=scope)
        else:
            return inception_preprocessing.preprocess_for_eval(image, height, width, central_fraction=None)


class inception_preprocess_with_rot:
    @staticmethod
    def preprocess_image(image, height, width,
                     is_training=False,
                     bbox=None,
                     fast_mode=True,
                     scope=None):
        return preprocess_image_base(image, height, width,
                                     is_training=is_training,
                                     bbox=bbox,
                                     fast_mode=fast_mode,
                                     scope=scope,
                                     distort_color=False,
                                     crop_area_range=[0.05, 1.0],
                                     rot_mode=True)


def get_edge(image):
    with tf.name_scope('get_edge', 'get_edge', [image]):
        conv_kernal = [[[[0.5, 0.5]], [[-1., 1.]], [[0.5, 0.5]]],
                       [[[1., -1.]], [[-2., -2.]], [[1., -1.]]],
                       [[[0.5, 0.5]], [[-1., 1.]], [[0.5, 0.5]]]]
        image_edge = tf.expand_dims(tf.image.rgb_to_grayscale(image), 0)
        # image = tf.cast(image, tf.float32)
        image_edge = tf.nn.conv2d(image_edge, conv_kernal, [1, 1, 1, 1], padding='SAME')

        image_mask = tf.cast(image_edge > 0, tf.float32)
        image_edge = image_edge * (image_mask * 2.0 - 1.0)

        image_edge = tf.reduce_sum(image_edge, 3, keep_dims=True)
        print("image edge shape: ", image_edge.get_shape().as_list())
        tv.image_summary('image_edge', image_edge)
        image_edge = tf.squeeze(image_edge)
        return image_edge


def add_edge(image):
    with tf.name_scope('add_edge', 'add_edge', [image]):
        image_edge = get_edge(image)
        image_r, image_g, image_b = tv.unpack(image, 3, axis=2)
        image = tv.pack([image_r, image_g, image_b, image_edge], axis=2)
        return image


class inception_preprocess_expand_edge:
    @staticmethod
    def preprocess_image(image, height, width,
                         is_training=False,
                         bbox=None,
                         fast_mode=True,
                         scope=None):
        image = preprocess_image_base(image, height, width,
                                      is_training=is_training,
                                      bbox=bbox,
                                      fast_mode=fast_mode,
                                      scope=scope,
                                      distort_color=False,
                                      crop_area_range=[0.05, 1.0],
                                      rot_mode=True)
        return add_edge(image)


class inception_preprocess_edge_no_crop:
    @staticmethod
    def preprocess_image(image, height, width,
                         is_training=False,
                         bbox=None,
                         fast_mode=True,
                         scope=None):
        if is_training:
            image = inception_preprocessing.preprocess_for_train(image, height, width, bbox, fast_mode, scope=scope)
        else:
            image = inception_preprocessing.preprocess_for_eval(image, height, width, central_fraction=None)

        return add_edge(image)


class inception_preprocess_only_edge:
    @staticmethod
    def preprocess_image(image, height, width,
                         is_training=False,
                         bbox=None,
                         fast_mode=True,
                         scope=None):
        image = preprocess_image_base(image, height, width,
                                      is_training=is_training,
                                      bbox=bbox,
                                      fast_mode=fast_mode,
                                      scope=scope,
                                      distort_color=False,
                                      crop_area_range=[0.05, 1.0],
                                      rot_mode=True)
        image = get_edge(image)
        image = tf.expand_dims(image, 2)
        return image


class inception_preprocess_only_edge_no_crop:
    @staticmethod
    def preprocess_image(image, height, width,
                         is_training=False,
                         bbox=None,
                         fast_mode=True,
                         scope=None):
        if is_training:
            image = inception_preprocessing.preprocess_for_train(image, height, width, bbox, fast_mode, scope=scope)
        else:
            image = inception_preprocessing.preprocess_for_eval(image, height, width, central_fraction=None)

        image = get_edge(image)
        image = tf.expand_dims(image, 2)
        return image


def add_normal_noise(image, height, width, channel=3, noise_percent=0.5, stddev=0.33):
    normal_noise = tf.random_normal([height, width, channel], stddev=stddev)
    image = normal_noise*noise_percent \
            + image*(tf.constant(1.0, dtype=tf.float32) - noise_percent)
    return image


def random_add_normal_noise(image, height, width,
                            channel=3, max_percent=0.5, max_dev=0.5, seed=None,
                            add_percent=0.5, name=None):
    with tf.name_scope(name, 'random_noise', [image, seed]) as scope:
        dice_random = tf.random_uniform([], 0, 1, dtype=tf.float32, seed=seed)
        precent_random = tf.random_uniform([], 0, max_percent, dtype=tf.float32, seed=seed)
        stddev_random = tf.random_uniform([], 0, max_dev, dtype=tf.float32, seed=seed)

        def add_noise():
            return add_normal_noise(image, height, width, channel=channel,
                                    noise_percent=precent_random, stddev=stddev_random)
        def do_nothing():
            return image

        return tf.cond(tf.less_equal(dice_random, add_percent), add_noise, do_nothing)


def get_gauss_kernal(top=3, bottom=3, left=3, right=3, stddev=2.0):
    size_h = left+right+1
    size_v = top+bottom+1
    arr_h = tf.cast(tf.reshape(tf.range(size_h), [size_h, 1]) * tf.ones([1, size_v], dtype=tf.int32) - left, tf.float32)
    arr_v = tf.cast(tf.reshape(tf.range(size_v), [1, size_v]) * tf.ones([size_h, 1], dtype=tf.int32) - top, tf.float32)
    gf = tf.exp(-(arr_h**2+arr_v**2)/(2*(stddev**2))) / (2*3.14*(stddev**2))
    gf = gf / tf.reduce_sum(gf)
    gf = tf.reshape(gf, [size_h, size_v, 1, 1])
    return gf


def add_gauss_filter(image, top=3, bottom=3, left=3, right=3, stddev=2.0, name=None):
    with tf.name_scope(name, 'gauss_filter', [image]) as scope:
        image = tf.transpose(image, perm=[2, 0, 1])
        image = tf.expand_dims(image, 3)
        gk = get_gauss_kernal(top=top, bottom=bottom, right=right, left=left, stddev=stddev)
        image = tf.nn.conv2d(image, gk, [1, 1, 1, 1], padding='SAME')
        image = tf.squeeze(image, [3])
        image = tf.transpose(image, perm=[1, 2, 0])
        return image


def random_add_gauss_filter(image, max_r=5, max_dev=50.0, seed=None, add_percent=0.5, name=None):
    with tf.name_scope(name, 'random_noise', [image, seed]) as scope:
        dice_random = tf.random_uniform([], 0, 1, dtype=tf.float32, seed=seed)
        rand_r = tf.random_uniform([4], 0, max_r, dtype=tf.int32, seed=seed)
        top, bottom, left, right = tv.unpack(rand_r, 4)
        stddev_random = tf.random_uniform([], 0, max_dev, dtype=tf.float32, seed=seed)

        def add_filter():
            return add_gauss_filter(image, top=top, bottom=bottom, left=left, right=right, stddev=stddev_random)

        def do_nothing():
            return image

        return tf.cond(tf.less_equal(dice_random, add_percent), add_filter, do_nothing)


class inception_preprocess_add_noise:
    @staticmethod
    def preprocess_image(image, height, width,
                         is_training=False,
                         bbox=None,
                         fast_mode=True,
                         scope=None):
        if is_training:
            image = preprocess_for_train(image, height, width, bbox, fast_mode, scope=scope,
                                         aspect_ratio_range=(0.5, 2.0))
            image = random_add_gauss_filter(image)
            tv.image_summary('image_with_gauss_filter', tf.expand_dims(image, 0))
            image = random_add_normal_noise(image, height, width)
            tv.image_summary('image_with_normal_noise', tf.expand_dims(image, 0))
        else:
            image = inception_preprocessing.preprocess_for_eval(image, height, width, central_fraction=None)
        return image


class inception_preprocess_add_noise_edge:
    @staticmethod
    def preprocess_image(image, height, width,
                         is_training=False,
                         bbox=None,
                         fast_mode=True,
                         scope=None):
        if is_training:
            image = preprocess_for_train(image, height, width, bbox, fast_mode, scope=scope,
                                         aspect_ratio_range=(0.5, 2.0))
            image = random_add_gauss_filter(image)
            tv.image_summary('image_with_gauss_filter', tf.expand_dims(image, 0))
            image = random_add_normal_noise(image, height, width)
            tv.image_summary('image_with_normal_noise', tf.expand_dims(image, 0))
        else:
            image = inception_preprocessing.preprocess_for_eval(image, height, width, central_fraction=None)
        return add_edge(image)


def get_line(matrix_x, matrix_y, seed=None, max_r=5):
    rand_val = tf.random_uniform([7], -1, 1, dtype=tf.float32, seed=seed)
    line_a, line_b, line_c, line_r, color_r, color_g, color_b = tv.unpack(rand_val, 7)
    line_r = tf.abs((line_a+line_b) * max_r * line_r)

    matrix_val = line_a*matrix_x + line_b*matrix_y + line_c*matrix_x.get_shape().as_list()[0]
    matrix_one = tf.logical_and(tf.less_equal(matrix_val-line_r, 0), tf.greater_equal(matrix_val+line_r, 0))
    matrix_one = tf.cast(matrix_one, tf.float32)

    matrix_col = tv.pack([matrix_one*color_r, matrix_one*color_g, matrix_one*color_b], axis=2)
    return matrix_col


def test_get_line(height=400, width=400):
    matrix_x = tf.ones([height, 1], dtype=tf.float32) * tf.reshape(tf.cast(tf.range(0, width), tf.float32), [1, width])
    matrix_y = tf.reshape(tf.cast(tf.range(0, height), tf.float32), [height, 1]) * tf.ones([1, width], dtype=tf.float32)

    img = get_line(matrix_x, matrix_y)
    img = img+get_line(matrix_x, matrix_y)
    img = img+get_line(matrix_x, matrix_y)
    img = img+get_line(matrix_x, matrix_y)
    img = img+get_line(matrix_x, matrix_y)
    img_sum = tf.reduce_sum(img)
    img = (img + 1) / 2 * 255
    img = tf.cast(img, tf.uint8)
    img_f = tf.image.encode_jpeg(img)

    sess = tf.Session()

    c, s = sess.run([img_f, img_sum])
    print("sum=%f" % s)
    with open('line.jpg', 'w') as f:
        f.write(c)


def add_line(image, height, width, line_num=10):
    matrix_x = tf.ones([height, 1], dtype=tf.float32) * tf.reshape(tf.cast(tf.range(0, width), tf.float32), [1, width])
    matrix_y = tf.reshape(tf.cast(tf.range(0, height), tf.float32), [height, 1]) * tf.ones([1, width], dtype=tf.float32)

    for i in range(line_num):
        image = image + get_line(matrix_x, matrix_y)

    image = tf.maximum(image, -1)
    image = tf.minimum(image, 1)

    return image


def random_add_line(image, height, width, line_num=10, add_percent=0.5):
    dice_random = tf.random_uniform([], 0, 1, dtype=tf.float32, seed=None)

    def do_add_line():
        return add_line(image, height, width, line_num)

    def do_nothing():
        return image

    return tf.cond(tf.less_equal(dice_random, add_percent), do_add_line, do_nothing)


def get_point(matrix_x, matrix_y, seed=None, max_r=5):
    rand_val = tf.random_uniform([6], -1, 1, dtype=tf.float32, seed=seed)
    point_x, point_y, point_r, color_r, color_g, color_b = tv.unpack(rand_val, 6)
    point_x = tf.abs(point_x) * matrix_x.get_shape().as_list()[1]
    point_y = tf.abs(point_y) * matrix_y.get_shape().as_list()[0]
    point_r = tf.abs(point_r)

    matrix_val = (matrix_x-point_x)**2 + (matrix_y-point_y)**2 - (max_r*point_r)**2
    matrix_one = tf.less_equal(matrix_val, 0)
    matrix_one = tf.cast(matrix_one, tf.float32)

    matrix_col = tv.pack([matrix_one * color_r, matrix_one * color_g, matrix_one * color_b], axis=2)
    return matrix_col


def test_get_point(height=400, width=400):
    matrix_x = tf.ones([height, 1], dtype=tf.float32) * tf.reshape(tf.cast(tf.range(0, width), tf.float32), [1, width])
    matrix_y = tf.reshape(tf.cast(tf.range(0, height), tf.float32), [height, 1]) * tf.ones([1, width], dtype=tf.float32)

    img = get_point(matrix_x, matrix_y)
    img_sum = tf.reduce_sum(img)
    img = (img + 1) / 2 * 255
    img = tf.cast(img, tf.uint8)
    img_f = tf.image.encode_jpeg(img)

    sess = tf.Session()

    c, s = sess.run([img_f, img_sum])
    print("sum=%f" % s)
    with open('point.jpg', 'w') as f:
        f.write(c)


def add_point(image, height, width, point_num=5):
    matrix_x = tf.ones([height, 1], dtype=tf.float32) * tf.reshape(tf.cast(tf.range(0, width), tf.float32), [1, width])
    matrix_y = tf.reshape(tf.cast(tf.range(0, height), tf.float32), [height, 1]) * tf.ones([1, width], dtype=tf.float32)

    for i in range(point_num):
        image = image + get_point(matrix_x, matrix_y)

    image = tf.maximum(image, -1)
    image = tf.minimum(image, 1)

    return image


def random_add_point(image, height, width, point_num=5, add_percent=0.5):
    dice_random = tf.random_uniform([], 0, 1, dtype=tf.float32, seed=None)

    def do_add_point():
        return add_point(image, height, width, point_num)

    def do_nothing():
        return image

    return tf.cond(tf.less_equal(dice_random, add_percent), do_add_point, do_nothing)


def random_rot(image, max_degree=10):
    rand_degree = tf.random_uniform([], -max_degree, max_degree, dtype=tf.float32, name='rand_degree')
    img_rot = tf.contrib.image.rotate(image, rand_degree * math.pi / 180.0)
    return img_rot


class inception_preprocess_add_noise_line:
    @staticmethod
    def preprocess_image(image, height, width,
                         is_training=False,
                         bbox=None,
                         fast_mode=True,
                         scope=None,
                         **unused_args):
        if is_training:
            image = preprocess_for_train(image, height, width, bbox, fast_mode, scope=scope,
                                         aspect_ratio_range=(0.5, 2.0))
            image = add_line(image, height, width)
            tv.image_summary('image_with_add_line', tf.expand_dims(image, 0))
            image = add_point(image, height, width)
            tv.image_summary('image_with_add_point', tf.expand_dims(image, 0))
            image = random_add_gauss_filter(image)
            tv.image_summary('image_with_gauss_filter', tf.expand_dims(image, 0))
            image = random_add_normal_noise(image, height, width)
            tv.image_summary('image_with_normal_noise', tf.expand_dims(image, 0))
        else:
            image = inception_preprocessing.preprocess_for_eval(image, height, width, central_fraction=None)
        return image


class inception_preprocess_resize_area:
    @staticmethod
    def preprocess_image(image, height, width,
                         is_training=False,
                         bbox=None,
                         fast_mode=True,
                         scope=None,
                         **unused_args):
        if is_training:
            return inception_preprocessing.preprocess_for_train(image, height, width, bbox, True, scope=scope)
        else:
            return preprocess_for_eval(image, height, width,
                                       resize_method=tf.image.ResizeMethod.AREA,
                                       central_fraction=None)


class inception_preprocess_not_fast:
    @staticmethod
    def preprocess_image(image, height, width,
                         is_training=False,
                         bbox=None,
                         scope=None,
                         **unused_args):
        if is_training:
            return inception_preprocessing.preprocess_for_train(image, height, width, bbox,
                                                                fast_mode=False, scope=scope)
        else:
            return preprocess_for_eval(image, height, width, central_fraction=None)


class inception_preprocess_all:
    @staticmethod
    def preprocess_image(image, height, width,
                         is_training=False,
                         bbox=None,
                         scope=None,
                         label=None,
                         special_labels_list=None,
                         **unused_args):
        if is_training:
            #image = preprocess_for_train(image, height, width, bbox, False, scope=scope,
            #                             aspect_ratio_range=(0.5, 2.0), crop_area_range=(0.81, 1))

            tv.image_summary('image_original', tf.expand_dims(image, 0))

            # 对于指定的labelid，crop时使用特殊的参数
            tensor_special_labels_list = tf.convert_to_tensor(special_labels_list, dtype=tf.int64)
            tf.logging.info("************************** tensor_special_labels_list: %s", tensor_special_labels_list)

            def normal_image_preprocessing_fn():
                return preprocess_for_train(image, height, width, bbox, False, scope=scope,
                                            aspect_ratio_range=(0.5, 2.0), crop_area_range=(0.05, 1))

            def special_image_preprocessing_fn():
                return preprocess_for_train(image, height, width, bbox, False, scope=scope,
                                            aspect_ratio_range=(0.5, 2.0), crop_area_range=(0.81, 1))

            image = tf.cond(tf.equal(tf.reduce_sum(tf.cast(
                tf.equal(tensor_special_labels_list, tf.ones(tensor_special_labels_list.shape, dtype=tf.int64) * label),
                tf.int64)), 1), special_image_preprocessing_fn, normal_image_preprocessing_fn)

            tv.image_summary('image_with_basic_preprocess', tf.expand_dims(image, 0))

            image = random_rot(image)
            tv.image_summary('image_with_rotate', tf.expand_dims(image, 0))
            image = random_add_line(image, height, width)
            tv.image_summary('image_with_add_line', tf.expand_dims(image, 0))
            image = random_add_point(image, height, width)
            tv.image_summary('image_with_add_point', tf.expand_dims(image, 0))
            image = random_add_gauss_filter(image)
            tv.image_summary('image_with_gauss_filter', tf.expand_dims(image, 0))
            image = random_add_normal_noise(image, height, width)
            tv.image_summary('image_with_normal_noise', tf.expand_dims(image, 0))

            return image
        else:
            return preprocess_for_eval(image, height, width, central_fraction=None)


class inception_preprocess_common_ori:
    @staticmethod
    def preprocess_image(image, height, width,
                         is_training=False,
                         bbox=None,
                         scope=None,
                         label=None,
                         special_labels_list=None,
                         crop_area_min=0.05,
                         **unused_args):
        if is_training:
            tf.logging.info("******* inception_preprocess_common crop_area_min: %f", crop_area_min)
            return preprocess_for_train(image, height, width, bbox, False, scope=scope,
                                            aspect_ratio_range=(0.56, 1.78), crop_area_range=(crop_area_min, 1))
        else:
            return preprocess_for_eval(image, height, width, central_fraction=None)
            

class inception_preprocess_common:
    @staticmethod
    def preprocess_image(image, height, width,
                         is_training=False,
                         bbox=None,
                         scope=None,
                         label=None,
                         special_labels_list=None,
                         crop_area_min=0.05,
                         **unused_args):
        if is_training:

            # 对于指定的labelid，crop时使用特殊的参数
            tensor_special_labels_list = tf.convert_to_tensor(special_labels_list, dtype=tf.int64)
            tf.logging.info("******* inception_preprocess_common tensor_special_labels_list: %s",
                            tensor_special_labels_list)
            tf.logging.info("******* inception_preprocess_common crop_area_min: %f", crop_area_min)

            def normal_image_preprocessing_fn():
                return preprocess_for_train(image, height, width, bbox, False, scope=scope,
                                            aspect_ratio_range=(0.56, 1.78), crop_area_range=(crop_area_min, 1))

            def special_image_preprocessing_fn():
                return preprocess_for_train(image, height, width, bbox, False, scope=scope,
                                            aspect_ratio_range=(0.56, 1.78), crop_area_range=(0.81, 1))

            image = tf.cond(tf.equal(tf.reduce_sum(tf.cast(
                tf.equal(tensor_special_labels_list,
                         tf.ones(tensor_special_labels_list.shape, dtype=tf.int64) * label),
                tf.int64)), 1), special_image_preprocessing_fn, normal_image_preprocessing_fn)

            #image = random_rot(image)
            #tv.image_summary('image_with_rotate', tf.expand_dims(image, 0))
            #image = add_line(image, height, width)

            return image
        else:
            return preprocess_for_eval(image, height, width, central_fraction=None)


class inception_preprocess_common_min_crop:
    @staticmethod
    def preprocess_image(image, height, width,
                         is_training=False,
                         bbox=None,
                         scope=None,
                         label=None,
                         special_labels_list=None,
                         crop_area_min=0.05,
                         is_crop=None,
                         **unused_args):
        if is_training:

            # 对于指定的labelid，crop时使用特殊的参数
            tensor_special_labels_list = tf.convert_to_tensor(special_labels_list, dtype=tf.int64)
            tf.logging.info("******* inception_preprocess_common tensor_special_labels_list: %s",
                            tensor_special_labels_list)
            tf.logging.info("******* inception_preprocess_common crop_area_min: %f", crop_area_min)

            def normal_image_preprocessing_fn():
                return preprocess_for_train(image, height, width, bbox, False, scope=scope,
                                            aspect_ratio_range=(0.56, 1.78), crop_area_range=(crop_area_min, 1))

            def special_image_preprocessing_fn():
                return preprocess_for_train(image, height, width, bbox, False, scope=scope,
                                            aspect_ratio_range=(0.56, 1.78), crop_area_range=(0.9, 1))

            image = tf.cond(tf.equal(is_crop, tf.constant(1, dtype=tf.int64)),
                            special_image_preprocessing_fn, normal_image_preprocessing_fn)

            #image = random_rot(image)
            #tv.image_summary('image_with_rotate', tf.expand_dims(image, 0))
            #image = add_line(image, height, width)

            return image
        else:
            return preprocess_for_eval(image, height, width, central_fraction=None)


class inception_preprocess_common_min_crop_mul_size:
    @staticmethod
    def preprocess_image(image, size_L, size_S,
                         is_training=False,
                         bbox=None,
                         scope=None,
                         label=None,
                         special_labels_list=None,
                         crop_area_min=0.05,
                         is_crop=None,
                         **unused_args):
        if is_training:

            # 对于指定的labelid，crop时使用特殊的参数
            tensor_special_labels_list = tf.convert_to_tensor(special_labels_list, dtype=tf.int64)
            tf.logging.info("******* inception_preprocess_common tensor_special_labels_list: %s",
                            tensor_special_labels_list)
            tf.logging.info("******* inception_preprocess_common crop_area_min: %f", crop_area_min)

            def normal_image_preprocessing_fn():
                return preprocess_for_train_mul_size(image, size_L, size_S, bbox, False, scope=scope,
                                            aspect_ratio_range=(0.5, 1.0), crop_area_range=(crop_area_min, 1))

            def special_image_preprocessing_fn():
                return preprocess_for_train_mul_size(image, size_L, size_S, bbox, False, scope=scope,
                                            aspect_ratio_range=(0.5, 1.0), crop_area_range=(0.9, 1))

            imageL, imageS = tf.cond(tf.equal(is_crop, tf.constant(1, dtype=tf.int64)),
                            special_image_preprocessing_fn, normal_image_preprocessing_fn)

            #image = random_rot(image)
            #tv.image_summary('image_with_rotate', tf.expand_dims(image, 0))
            #image = add_line(image, height, width)

            return imageL, imageS
        else:
            return preprocess_for_eval(image, size_S, size_S, central_fraction=None)
