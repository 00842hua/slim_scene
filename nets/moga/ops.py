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


def Hswish(features):
    """ hard swish"""
    return features * tf.nn.relu6(features+3) / 6.0

def Hsigmoid(features):
    """ hard sigmoid"""
    return tf.nn.relu6(features+3) / 6.0


def conv2d_bn(input, out_dim, k, s, is_train, scope, non_linear="RE"):
    with tf.variable_scope(scope):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'is_training': is_train,
        }

        activation = tf.nn.relu if non_linear=='RE' else None
        if s == 1:
            net = slim.conv2d(input, num_outputs=out_dim, kernel_size=k, stride=s, activation_fn=activation,
                            normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params)
        else:
            padding = k // 2
            input = tf.pad(input, paddings=[[0, 0], [padding, padding], [padding, padding], [0, 0]])
            net = slim.conv2d(input,
                              num_outputs=out_dim,
                              kernel_size=k,
                              stride=s,
                              activation_fn=activation,
                              padding="VALID",
                              normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params)
            
        if non_linear == "HS":
            net = Hswish(net)
        
        return net


def conv2d(input, out_dim, k, s, scope, non_linear='RE'):
    with tf.variable_scope(scope):
        activation = tf.nn.relu if non_linear=='RE' else None
        if s == 1:
            net = slim.conv2d(input, num_outputs=out_dim, kernel_size=k, stride=s, activation_fn=activation)
        else:
            padding = k // 2
            input = tf.pad(input, paddings=[[0, 0], [padding, padding], [padding, padding], [0, 0]])
            net = slim.conv2d(input,
                              num_outputs=out_dim,
                              kernel_size=k,
                              stride=s,
                              activation_fn=activation,
                              padding="VALID",
                              biases_initializer=None
                              )
        # net = slim.fully_connected(input, out_dim)
        
        if non_linear == "HS":
            net = Hswish(net)
        
        return net

def conv2d_bias(input, out_dim, k, s, scope, non_linear='RE'):
    with tf.variable_scope(scope):
        activation = tf.nn.relu if non_linear=='RE' else None
        if s == 1:
            net = slim.conv2d(input, num_outputs=out_dim, kernel_size=k, stride=s, activation_fn=activation,
                            normalizer_fn=None, biases_initializer=tf.zeros_initializer())
        else:
            padding = k // 2
            input = tf.pad(input, paddings=[[0, 0], [padding, padding], [padding, padding], [0, 0]])
            net = slim.conv2d(input,
                              num_outputs=out_dim,
                              kernel_size=k,
                              stride=s,
                              activation_fn=activation,
                              padding="VALID",
                              biases_initializer=tf.zeros_initializer()
                              )

        # net = slim.fully_connected(input, out_dim)
        
        if non_linear == "HS":
            net = Hswish(net)
        
        return net

def squeeze(input, channel, scope='squeeze', non_linear='RE'):
    with tf.variable_scope(scope):
        _, in_h, in_w, in_dim = input.shape
        activation = tf.nn.relu if non_linear=='RE' else None
        # print('Squeeze:', in_h, in_w, channel)
        net = slim.avg_pool2d(input, kernel_size=(in_h,in_w))
        net = slim.conv2d(net, num_outputs=channel, kernel_size=1, stride=1, activation_fn=activation)
        if non_linear == 'HS':
            net = Hswish(net)
        net = slim.conv2d(net, num_outputs=in_dim, kernel_size=1, stride=1, activation_fn=None)
        net = Hsigmoid(net)
        net = tf.multiply(input, net)
        return net


def res_block(input, expansion_ratio, output_dim, kernel_size, stride, is_train, scope, non_linear, squeeze_excitation=False, use_conv_for_expansion=True):
    with tf.variable_scope(scope):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'is_training': is_train,
        }
        _, _, _, in_dim = input.shape
        in_dim = int(in_dim)
        activation = tf.nn.relu if non_linear=='RE' else None
        
        if use_conv_for_expansion:
            # expansion
            bottleneck_dim = round(expansion_ratio * input.get_shape().as_list()[-1])
            net = slim.conv2d(input, num_outputs=bottleneck_dim, kernel_size=1, stride=1, activation_fn=activation,
                            normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, scope="pw1")

            if non_linear == 'HS':
                net = Hswish(net)
        else:
            # Song Han's Proxyless exclude first conv
            net = input
        
        # depthwise
        if stride == 1:
            net = slim.separable_conv2d(net, num_outputs=None, kernel_size=kernel_size, depth_multiplier=1,
                                    stride=stride, activation_fn=activation,
                                    normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, scope="dw")
        else:
            padding = kernel_size // 2
            input = tf.pad(net, paddings=[[0, 0], [padding, padding], [padding, padding], [0, 0]])
            net = slim.separable_conv2d(input,
                              num_outputs=None,
                              kernel_size=kernel_size,
                              depth_multiplier=1,
                              stride=stride,
                              activation_fn=activation,
                              normalizer_fn=slim.batch_norm, 
                              normalizer_params=batch_norm_params,
                              padding="VALID",
                              biases_initializer=None, 
                              scope="dw",
                              )
        
        if non_linear == 'HS':
            net = Hswish(net)
        
        if squeeze_excitation:
            net = squeeze(net, scope='squeeze', channel=round(bottleneck_dim/4), non_linear=non_linear)

        # pw & linear
        net = slim.conv2d(net, num_outputs=output_dim, kernel_size=1, stride=1, activation_fn=None,
                          normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, scope="pw2")
        # res
        if in_dim == output_dim and stride == 1:
            net = net + input
            return net
        else:
            return net


def separable_conv(input, output_dim, kernel_size, stride, is_train, scope):
    with tf.variable_scope(scope):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'is_training': is_train,
        }
        net = slim.separable_conv2d(input, num_outputs=None, kernel_size=kernel_size, depth_multiplier=1,
                                    stride=stride, activation_fn=tf.nn.relu,
                                    normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, scope="dw")
        net = slim.conv2d(net, num_outputs=output_dim, kernel_size=1, stride=1, activation_fn=None,
                          normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, scope="pw")
        return net

