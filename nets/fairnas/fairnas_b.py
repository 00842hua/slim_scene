from nets.fairnas.tf_ops import *
from nets.fairnas.SearchSpace import get_param

import tensorflow as tf
import tensorflow.contrib.slim as slim


def fairnas_b(inputs, num_classes, is_training=False):
    model_meta = [1, 0, 1, 0, 3, 1, 2, 0, 3, 1, 0, 3, 2, 0, 5, 4, 5, 3, 4]
    with slim.arg_scope(global_scope(is_training=is_training)):
        with tf.variable_scope("FairNas", values=[inputs]):
            end_points = {}
            net = conv2d_bn(inputs, 32, 3, 2, scope='stem')
            net = separable_conv(net, 16, 3, 1, scope='separable_conv')
            for index, id in enumerate(model_meta):
                expansion, channel, kernel_size, stride = get_param(index, id)
                net = res_block(net, expansion, channel, kernel_size, stride, scope="mb_%d" % index)
            net = conv2d_bn(net, 1280, 1, 1, scope='conv_before_pooling')
            net = tf.layers.average_pooling2d(net, [7,7], 1)
            net = conv2d_bias(net, num_classes, 1, 1, activation_fn=None, scope="conv_after_pooling")
            logits = tf.squeeze(net, axis=[1, 2])
            end_points['Logits'] = logits
            end_points['Predictions_sigmoid'] = tf.nn.sigmoid(logits, name='Predictions_sigmoid')
            end_points['Predictions_softmax'] = tf.nn.softmax(logits, name='Predictions_softmax')
            return logits, end_points
