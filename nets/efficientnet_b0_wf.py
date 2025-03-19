# -*- coding: utf-8 -*-

"""Model Builder for EfficientNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

__all__ = ['efficientnet_b0_wf']

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _batch_normalization_layer(inputs, momentum=0.997, epsilon=1e-3, is_training=True, name='bn', reuse=None):
    return tf.layers.batch_normalization(inputs=inputs,
                                         momentum=momentum,
                                         epsilon=epsilon,
                                         scale=True,
                                         center=True,
                                         training=is_training,
                                         name=name,
                                         reuse=reuse)


def _conv2d_layer(inputs, filters_num, kernel_size, name, use_bias=False, strides=1, reuse=None, padding="SAME"):
    conv = tf.layers.conv2d(
        inputs=inputs, filters=filters_num,
        kernel_size=kernel_size, strides=[strides, strides], kernel_initializer=tf.glorot_uniform_initializer(),
        padding=padding, #('SAME' if strides == 1 else 'VALID'),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4), use_bias=use_bias, name=name,
        reuse=reuse)
    return conv


def _conv_1x1_bn(inputs, filters_num, name, use_bias=True, is_training=True, reuse=None):
    kernel_size = 1
    strides = 1
    x = _conv2d_layer(inputs, filters_num, kernel_size, name=name + "/conv", use_bias=use_bias, strides=strides)
    x = _batch_normalization_layer(x, momentum=0.997, epsilon=1e-3, is_training=is_training, name=name + '/bn',
                                   reuse=reuse)
    return x


def _conv_1x1(inputs, filters_num, name, use_bias=True):
    kernel_size = 1
    strides = 1
    x = _conv2d_layer(inputs, filters_num, kernel_size, name=name + "/conv", use_bias=use_bias, strides=strides)

    return x


def _conv_bn_relu(inputs, filters_num, kernel_size, name, use_bias=True, strides=1, is_training=True,
                  activation=tf.nn.relu6, reuse=None):
    x = _conv2d_layer(inputs, filters_num, kernel_size, name, use_bias=use_bias, strides=strides)
    x = _batch_normalization_layer(x, momentum=0.997, epsilon=1e-3, is_training=is_training, name=name + '/bn',
                                   reuse=reuse)
    x = activation(x)
    return x


def _dwise_conv(inputs, k_h=3, k_w=3, depth_multiplier=1, strides=(1, 1),
                padding='SAME', name='dwise_conv', use_bias=False,
                reuse=None):
    kernel_size = (k_w, k_h)
    in_channel = inputs.get_shape().as_list()[-1]
    filters = int(in_channel*depth_multiplier)
    return tf.layers.separable_conv2d(inputs, filters, kernel_size,
                                      strides=strides, padding=padding,
                                      data_format='channels_last', dilation_rate=(1, 1),
                                      depth_multiplier=depth_multiplier, activation=None,
                                      use_bias=use_bias, name=name, reuse=reuse
                                      )


def relu6(x, name='relu6'):
    return tf.nn.relu6(x, name)


def hard_swish(x, name='hard_swish'):
    with tf.variable_scope(name):
        h_swish = x * tf.nn.relu6(x + 3) / 6
    return h_swish


def hard_sigmoid(x, name='hard_sigmoid'):
    with tf.variable_scope(name):
        h_sigmoid = tf.nn.relu6(x + 3) / 6
    return h_sigmoid


def _fully_connected_layer(inputs, units, name="fc", activation=None, use_bias=True, reuse=None):
    return tf.layers.dense(inputs, units, activation=activation, use_bias=use_bias,
                           name=name, reuse=reuse)


def _global_avg(inputs, pool_size, strides, padding='valid', name='global_avg'):
    return tf.layers.average_pooling2d(inputs, pool_size, strides,
                                       padding=padding, data_format='channels_last', name=name)


def _squeeze_excitation_layer(input, out_dim, se_dim, layer_name, is_training=True, reuse=None):
    with tf.variable_scope(layer_name, reuse=reuse):
        squeeze = _global_avg(input, pool_size=input.get_shape()[1:-1], strides=1)

        #excitation = _fully_connected_layer(squeeze, units=se_dim, name=layer_name + '_excitation1', reuse=reuse)
        excitation = _conv_1x1(squeeze, se_dim, name=layer_name + '_excitation1')
        tf.logging.info('----- se shape: %s' % (excitation.shape))
        excitation = relu6(excitation)
        #excitation = _fully_connected_layer(excitation, units=out_dim, name=layer_name + '_excitation2', reuse=reuse)
        excitation = _conv_1x1(excitation, out_dim, name=layer_name + '_excitation2')
        excitation = hard_sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        tf.logging.info('----- excitation shape: %s' % (excitation.shape))
        scale = input * excitation
        tf.logging.info('----- after se shape: %s' % (scale.shape))
        return scale


def mobilenet_v3_block(input, k_s, expansion_ratio, output_dim, stride, name, is_training=True,
                       use_bias=True, shortcut=True, activatation="RE", ratio=4, se=False,
                       reuse=None):
    bottleneck_dim = expansion_ratio
    indim = input.get_shape().as_list()[-1]
    tf.logging.info('-------------------------- mobilenet_v3_block %s' % (name))
    tf.logging.info('input_shape(%s) kernal_size(%s) expansion_ratio(%s) output_dim(%s) stride(%s)' \
                    % (input.shape, k_s, expansion_ratio, output_dim, stride))

    with tf.variable_scope(name, reuse=reuse):
        # pw mobilenetV2
        net = input
        if bottleneck_dim > indim:
            net = _conv_1x1_bn(net, bottleneck_dim, name="pw", use_bias=use_bias, is_training=is_training)

            tf.logging.info('expand shape: %s' % (net.shape))

            if activatation == "HS":
                net = hard_swish(net)
            elif activatation == "RE":
                net = relu6(net)
            elif activatation == "SW":
                net = tf.nn.swish(net)
            else:
                raise NotImplementedError

        # dw
        net = _dwise_conv(net, k_w=k_s, k_h=k_s, strides=[stride, stride], name='dw',
                          use_bias=use_bias, reuse=reuse)

        net = _batch_normalization_layer(net, momentum=0.997, epsilon=1e-3,
                                         is_training=is_training, name='dw_bn', reuse=reuse)

        if activatation == "HS":
            net = hard_swish(net)
        elif activatation == "RE":
            net = relu6(net)
        elif activatation == "SW":
            net = tf.nn.swish(net)
        else:
            raise NotImplementedError
        tf.logging.info('dw shape: %s' % (net.shape))
        # squeeze and excitation
        if se:
            channel = net.get_shape().as_list()[-1]
            se_dim = indim / ratio
            net = _squeeze_excitation_layer(net, out_dim=channel, se_dim=se_dim, layer_name='se_block')

            tf.logging.info('after se shape: %s' % (net.shape))

        # pw & linear
        net = _conv_1x1_bn(net, output_dim, name="pw_linear", use_bias=use_bias, is_training=is_training)
        tf.logging.info('pw shape: %s' % (net.shape))
        # element wise add, only for stride==1
        if shortcut and stride == 1 and indim == output_dim:
            net += input
            net = tf.identity(net, name='block_output')

    return net


def efficientnet_b0_wf(inputs, classes_num, multiplier=1.0, is_training=True, reuse=None):
    end_points = {}
    layers = [
        # (in_channels, out_channels, kernel_size, stride, activatation, se, exp_size)
        [32, 16, 3, 1, "SW", True, 1],  # 112*112*32 -> 112*112*16

        [16, 24, 3, 1, "SW", True, 6],  # 112*112*16 -> 112*112*24
        [24, 24, 3, 1, "SW", True, 6],

        [24, 40, 5, 2, "SW", True, 6],  # 56*56*40
        [40, 40, 5, 1, "SW", True, 6],

        [40, 80, 3, 2, "SW", True, 6],  # 28*28*80
        [80, 80, 3, 1, "SW", True, 6],
        [80, 80, 3, 1, "SW", True, 6],

        [80, 112, 5, 1, "SW", True, 6],  # 28*28*112
        [112, 112, 5, 1, "SW", True, 6],  # 28*28*112
        [112, 112, 5, 1, "SW", True, 6],  # 28*28*112

        [112, 192, 5, 2, "SW", True, 6],  # 14*14*192
        [192, 192, 5, 1, "SW", True, 6],
        [192, 192, 5, 1, "SW", True, 6],
        [192, 192, 5, 1, "SW", True, 6],

        [192, 320, 3, 2, "SW", True, 6],  # 7*7*320
    ]
    '''
    layers = [
        # (in_channels, out_channels, kernel_size, stride, activatation, se, exp_size)
        [32, 16, 3, 1, "SW", False, 1],  # 112*112*32 -> 112*112*16

        [16, 24, 3, 1, "SW", False, 6],  # 112*112*16 -> 112*112*24
        [24, 24, 3, 1, "SW", False, 6],

        [24, 40, 5, 2, "SW", False, 6],  # 56*56*40
        [40, 40, 5, 1, "SW", False, 6],

        [40, 80, 3, 2, "SW", False, 6],  # 28*28*80
        [80, 80, 3, 1, "SW", False, 6],
        [80, 80, 3, 1, "SW", False, 6],

        [80, 112, 5, 1, "SW", False, 6],  # 28*28*112
        [112, 112, 5, 1, "SW", False, 6],  # 28*28*112
        [112, 112, 5, 1, "SW", False, 6],  # 28*28*112

        [112, 192, 5, 2, "SW", False, 6],  # 14*14*192
        [192, 192, 5, 1, "SW", False, 6],
        [192, 192, 5, 1, "SW", False, 6],
        [192, 192, 5, 1, "SW", False, 6],

        [192, 320, 3, 2, "SW", False, 6],  # 7*7*320
    ]
    '''
    tf.logging.info('input shape: %s' % (inputs.shape))

    input_size = inputs.get_shape().as_list()[1:-1]
    assert ((input_size[0] % 32 == 0) and (input_size[1] % 32 == 0))

    reduction_ratio = 4
    with tf.variable_scope('init', reuse=reuse):
        init_conv_out = _make_divisible(32 * multiplier)
        x = _conv_bn_relu(inputs, filters_num=init_conv_out, kernel_size=3, name='init',
                          use_bias=False, strides=2, is_training=is_training, activation=tf.nn.swish)
        tf.logging.info('init shape: %s' % (x.shape))

    with tf.variable_scope("efficientnet_b0", reuse=reuse):
        for idx, (in_channels, out_channels, kernel_size, stride, activatation, se, exp_size) in enumerate(layers):
            in_channels = _make_divisible(in_channels * multiplier)
            out_channels = _make_divisible(out_channels * multiplier)
            exp_size = _make_divisible(exp_size * in_channels)
            x = mobilenet_v3_block(x, kernel_size, exp_size, out_channels, stride,
                                   "bneck{}".format(idx), is_training=is_training, use_bias=True,
                                   shortcut=(in_channels==out_channels), activatation=activatation,
                                   ratio=reduction_ratio, se=se)
            end_points["bneck{}".format(idx)] = x
            tf.logging.info('bneck_%s shape: %s' % (idx, x.shape))

        conv1_out = _make_divisible(1280 * multiplier)
        x = _conv_bn_relu(x, filters_num=conv1_out, kernel_size=1, name="conv1_out",
                          use_bias=True, strides=1, is_training=is_training, activation=hard_swish)

        #x = _squeeze_excitation_layer(x, out_dim=conv1_out, ratio=reduction_ratio, layer_name="conv1_out",
        #                             is_training=is_training, reuse=None)
        end_points["conv1_out_1x1"] = x
        tf.logging.info('-------------------------- END blocks')
        tf.logging.info('_conv_bn_relu shape: %s' % (x.shape))

        with tf.variable_scope('Logits', reuse=reuse):

            x = _global_avg(x, pool_size=x.get_shape()[1:-1], strides=1)
            x = hard_swish(x)
            end_points["global_pool"] = x

            #dropout_drop_prob = 0.2
            #x = tf.layers.dropout(x, rate=dropout_drop_prob, name='Dropout', training=is_training)

            x = _conv_bn_relu(x, filters_num=classes_num, kernel_size=1, name="conv3", use_bias=True, strides=1,
                              is_training=is_training, activation=hard_swish)
            logits = tf.layers.flatten(x)
            logits = tf.identity(logits, name='output')
            end_points["Logits"] = logits
            end_points['Predictions_softmax'] = tf.nn.softmax(logits, name='Predictions_softmax')
            end_points['Predictions_sigmoid'] = tf.nn.sigmoid(logits, name='Predictions_sigmoid')

    return logits, end_points


efficientnet_b0_wf.default_image_size = 224
