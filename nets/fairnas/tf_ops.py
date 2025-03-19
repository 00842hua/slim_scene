import tensorflow as tf
import tensorflow.contrib.slim as slim


def global_scope(is_training=True,
                 weight_decay=4e-5,
                 stddev=0.09,
                 bn_decay=0.997):
    batch_norm_params = {
        'decay': bn_decay,
        'epsilon': 1e-5,
        'scale': True,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'is_training': is_training,
    }
    if stddev < 0:
        weight_intitializer = slim.initializers.xavier_initializer()
    else:
        weight_intitializer = tf.truncated_normal_initializer(stddev=stddev)
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.separable_conv2d],
                        weights_initializer=weight_intitializer,
                        normalizer_fn=slim.batch_norm), \
         slim.arg_scope([slim.batch_norm], **batch_norm_params), \
         slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(weight_decay)), \
         slim.arg_scope([slim.separable_conv2d], weights_regularizer=None) as sc:
        return sc


def conv2d_bn(inputs, out_channel, kernel_size, stride, activation_fn=tf.nn.relu6, scope=None, padding_type='VALID'):
    with tf.variable_scope(scope):
        if stride == 1:
            net = slim.conv2d(inputs,
                              num_outputs=out_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              activation_fn=activation_fn,
                              biases_initializer=None,
                              scope="conv2d_bn")
        else:
            if padding_type == 'VALID':
                padding = kernel_size // 2
                net = tf.pad(inputs, paddings=[[0, 0], [padding, padding], [padding, padding], [0, 0]])
            elif padding_type == 'SAME':
                net = inputs
            else:
                assert False, "Invalid padding type"
            
            net = slim.conv2d(net,
                              num_outputs=out_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              activation_fn=activation_fn,
                              padding=padding_type,
                              biases_initializer=None,
                              scope="conv2d_bn"
                              )
        return net


def conv2d_bias(inputs, out_channel, kernel_size, stride, activation_fn=tf.nn.relu6, scope=None):
    with tf.variable_scope(scope):
        net = slim.conv2d(inputs,
                          num_outputs=out_channel,
                          kernel_size=kernel_size,
                          stride=stride,
                          activation_fn=activation_fn,
                          normalizer_fn=None,
                          biases_initializer=tf.zeros_initializer(),
                          biases_regularizer=None,
                          scope="conv2d_bias")
        return net


def res_block(inputs, expansion_ratio, out_channel, kernel_size, stride, activation_fn=tf.nn.relu, scope=None, padding_type='VALID'):
    with tf.variable_scope(scope):
        _, _, _, in_dim = inputs.shape
        in_dim = int(in_dim)
        # up pw
        bottleneck_dim = round(expansion_ratio * inputs.get_shape().as_list()[-1])
        net = slim.conv2d(inputs,
                          num_outputs=bottleneck_dim,
                          kernel_size=1,
                          stride=1,
                          activation_fn=activation_fn,
                          biases_initializer=None,
                          scope="pw1")
        # dw
        if stride == 1:
            net = slim.separable_conv2d(net,
                                        num_outputs=None,
                                        kernel_size=kernel_size,
                                        depth_multiplier=1,
                                        stride=stride,
                                        activation_fn=activation_fn,
                                        biases_initializer=None,
                                        scope="dw")
        else:
            if padding_type == 'VALID':
                padding = kernel_size // 2
                net = tf.pad(net, paddings=[[0, 0], [padding, padding], [padding, padding], [0, 0]])
            elif padding_type == 'SAME':
                pass
            else:
                assert False, "Invalid padding type"
            net = slim.separable_conv2d(net,
                                        num_outputs=None,
                                        kernel_size=kernel_size,
                                        depth_multiplier=1,
                                        stride=stride,
                                        padding=padding_type,
                                        activation_fn=activation_fn,
                                        biases_initializer=None,
                                        scope="dw")

        # down pw
        net = slim.conv2d(net,
                          num_outputs=out_channel,
                          kernel_size=1,
                          stride=1,
                          activation_fn=None,
                          biases_initializer=None,
                          scope="pw2")
        # res
        if in_dim == out_channel and stride == 1:
            net = net + inputs
        return net


def separable_conv(inputs, out_channel, kernel_size, stride, activation_fn=tf.nn.relu6, scope=None):
    with tf.variable_scope(scope):
        net = slim.separable_conv2d(inputs,
                                    num_outputs=None,
                                    kernel_size=kernel_size,
                                    depth_multiplier=1,
                                    stride=stride,
                                    activation_fn=activation_fn,
                                    biases_initializer=None,
                                    scope="dw",
                                    #normalizer_fn=None
                                    )
        net = slim.conv2d(net,
                          num_outputs=out_channel,
                          kernel_size=1,
                          stride=1,
                          activation_fn=None,
                          biases_initializer=None,
                          scope="pw")
        return net
