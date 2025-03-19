from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

if tf.__version__.startswith("1."):
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    stack = tf.stack
    unstack = tf.unstack
    pack = tf.stack
    unpack = tf.unstack
    multiply = tf.multiply
    mul = tf.multiply
    subtract = tf.subtract
    sub = tf.subtract
    per_image_standardization = tf.image.per_image_standardization
    per_image_whitening = tf.image.per_image_standardization
    SummaryWriter = tf.summary.FileWriter
    neg = tf.negative
    negative = tf.negative

    def concat(axis, values, name='concat'):
        return tf.concat(values, axis, name=name)

elif tf.__version__.startswith("0."):
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    stack = tf.pack
    unstack = tf.unpack
    pack = tf.pack
    unpack = tf.unpack
    multiply = tf.mul
    mul = tf.mul
    subtract = tf.sub
    sub = tf.sub
    per_image_standardization = tf.image.per_image_whitening
    per_image_whitening = tf.image.per_image_whitening
    SummaryWriter = tf.train.SummaryWriter
    neg = tf.neg
    negative = tf.neg

    def concat(axis, values, name='concat'):
        return tf.concat(axis, values, name=name)