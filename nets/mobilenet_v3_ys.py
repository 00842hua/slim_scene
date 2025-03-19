from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools
import time
import tensorflow as tf
from datetime import datetime

slim = tf.contrib.slim

def mobilenet_v3_arg_scope(weight_decay = 1e-5,
                    batch_norm_decay = 0.997,
                    batch_norm_epsilon = 1e-5,
                    batch_norm_scale = True,
                    dropout_keep_prob = 0.8,
                    is_training = True):

    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected, slim.separable_conv2d], 
            weights_initializer = slim.initializers.xavier_initializer(),   #tf.truncated_normal_initializer(stddev=stddev)
            activation_fn = None):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                    weights_regularizer=slim.l2_regularizer(weight_decay)):
                with slim.arg_scope([slim.separable_conv2d], weights_regularizer=None): 
                    with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
                        return arg_sc



#define activate function relu6, hard_swish, hard_sigmoid
def relu(x, name = 'relu'):
    return tf.nn.relu(x, name)

def relu6(x, name = 'relu6'):
    return tf.nn.relu6(x,name)

def hard_swish(x, name = 'hard_swish'):
    with tf.variable_scope(name):
        hswish = x * tf.nn.relu6(x + 3) / 6
    return hswish

def hard_sigmoid(x, name = 'hard_sigmoid'):
    with tf.variable_scope(name):
        hsigmoid = tf.nn.relu6(x + 3) / 6
    return hsigmoid

def conv2d_bn_hs(input, output_dim, kernel_size, stride, name, is_training = True):
    with tf.variable_scope(name):
        out = slim.conv2d(input, output_dim, [kernel_size, kernel_size], stride = stride, 
                    activation_fn = None, scope = 'conv')
        out = slim.batch_norm(out, is_training = is_training, scope = 'bn')
        out = hard_swish(out)
    return out

def conv2d_nbn_hs(input, output_dim, kernel_size, stride, name, is_training = True):
    with tf.variable_scope(name):
        out = slim.conv2d(input, output_dim, [kernel_size, kernel_size], stride = stride, 
                        activation_fn = None, scope = name)
        out = hard_swish(out)
    return out


def _se_layer(input, out_dim, layer_name, ratio = 4):
    with tf.variable_scope(layer_name):
        squeeze = slim.avg_pool2d(input, kernel_size = input.get_shape()[1: -1], 
                                    stride = 1, scope = 'avg_pool2d')
        excitation = slim.fully_connected(squeeze, int(out_dim / ratio), activation_fn = None, \
                                    scope = 'excitation1')
        excitation = relu(excitation) # activation_fn  relu or relu6?
        excitation = slim.fully_connected(excitation, out_dim, activation_fn = None, \
                                    scope = 'excitation2')
        excitation = hard_sigmoid(excitation)
        
        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = input * excitation
        return scale


def mobilenet_v3_block(input_tensor, kernel_size, expand_size, out_size, \
                        stride, activation, name, is_training = True, se = False):
    with tf.variable_scope(name):
        net = slim.conv2d(input_tensor, expand_size, [1, 1], stride = 1, 
                    activation_fn = None, scope = 'pointwise_1')
        net = slim.batch_norm(net, is_training = is_training, scope = 'bn_1')

        if activation == 'h_swish':
            net = hard_swish(net, name = 'hswish_1')
        elif activation == 'relu6':
            net = relu6(net, name = 'relu6_1')
        else:
            raise ValueError('Unknown activation_fn')

        net = slim.separable_conv2d(net, None, [kernel_size, kernel_size], stride = stride,
                                    depth_multiplier = 1, activation_fn = None, normalizer_fn = None,
                                    scope = 'separable')
        net = slim.batch_norm(net, is_training = is_training, scope = 'bn_2')
        
        if activation == 'h_swish':
            net = hard_swish(net, name = 'hswish_2')
        elif activation == 'relu6':
            net = relu6(net, name = 'relu6_2')
        else:
            raise ValueError('Unknown activation_fn')
        
        if se:
            channel = net.get_shape().as_list()[-1]
            net = _se_layer(net, channel, layer_name = 'se_block')

        net = slim.conv2d(net, out_size, [1, 1], stride = 1,
                activation_fn = None, scope = 'pointwise_2')
        net = slim.batch_norm(net, is_training = is_training, scope = 'bn_3')

        # shortcut
        if stride == 1 and input_tensor.get_shape() == out_size :
            net += input_tensor 
            net = tf.identity(net, name = 'block_output')
        
        return net

def mobilenetv3_large(input, num_classes, prediction_fn = tf.nn.sigmoid, is_training = True):
    end_points = {}

    with tf.variable_scope('mobilenetv3_large'):
        # conv init
        net = conv2d_bn_hs(input, 16, 3, 2, name = 'init_conv1_1', is_training = is_training)     # 224 3 -> 112 16 
        # bneck2 112 
        net = mobilenet_v3_block(net, 3,  16, 16, 1, 'relu6', 'bneck2_1',  is_training = is_training, se = False) #112 16
    
        # bneck3 56 
        net = mobilenet_v3_block(net, 3,  64, 24, 2, 'relu6', 'bneck3_1',  is_training = is_training, se = False) #112 16 -> 56 24
        net = mobilenet_v3_block(net, 3,  72, 24, 1, 'relu6', 'bneck3_2',  is_training = is_training, se = False) #56 24
    
        # bneck4 28
        net = mobilenet_v3_block(net, 5,  72, 40, 2, 'relu6', 'bneck4_1',  is_training = is_training, se = True)
        net = mobilenet_v3_block(net, 5, 120, 40, 1, 'relu6', 'bneck4_2',  is_training = is_training, se = True)
        net = mobilenet_v3_block(net, 5, 120, 40, 1, 'relu6', 'bneck4_3',  is_training = is_training, se = True)
    
        #bneck5
        net = mobilenet_v3_block(net, 3, 240, 80, 2, 'h_swish','bneck5_1',  is_training = is_training, se = False)
        net = mobilenet_v3_block(net, 3, 200, 80, 1, 'h_swish','bneck5_2',  is_training = is_training, se = False)
        net = mobilenet_v3_block(net, 3, 184, 80, 1, 'h_swish','bneck5_3',  is_training = is_training, se = False)
        net = mobilenet_v3_block(net, 3, 184, 80, 1, 'h_swish','bneck5_4',  is_training = is_training, se = False)
    
        net = mobilenet_v3_block(net, 3, 480, 112, 1, 'h_swish','bneck5_5',  is_training = is_training, se = True)
        net = mobilenet_v3_block(net, 3, 672, 112, 1, 'h_swish','bneck5_6',  is_training = is_training, se = True)
        net = mobilenet_v3_block(net, 5, 672, 160, 1, 'h_swish','bneck5_7',  is_training = is_training, se = True)
        #                                   112 or 160 ?   v2 set it to 160
        # bneck6
        net = mobilenet_v3_block(net, 5, 672, 160, 2, 'h_swish','bneck6_1',  is_training = is_training, se = True)
        net = mobilenet_v3_block(net, 5, 960, 160, 1, 'h_swish','bneck6_2',  is_training = is_training, se = True)

        net = conv2d_bn_hs(net, 960, 1, 1, name = 'conv_1', is_training = is_training) 

        net = slim.avg_pool2d(net, [7, 7], stride = 1, scope = 'avg_pool')
        net = conv2d_nbn_hs(net, 1280, 1, 1, name = 'linear_1', is_training = is_training)
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn = None, \
                        normalizer_fn = None, scope = 'linear_2')
        logits = tf.layers.flatten(net)
        logits = tf.identity(logits, name = 'output')
        end_points['Logits'] = logits
        if prediction_fn:
            end_points['Predictions'] = prediction_fn(logits, name='Predictions')
        return logits, end_points

mobilenetv3_large.default_image_size = 224

if __name__ == "__main__":

    inputs = tf.random_normal([1, 224, 224, 3])
    
    with slim.arg_scope(mobilenet_v3_arg_scope(is_training = False)):
        logits ,end_points= mobilenetv3_large(inputs, num_classes = 34, is_training = False)
   
    writer = tf.summary.FileWriter("./logs", graph = tf.get_default_graph())
    
    
    print("Layers")
    for k, v in end_points.items():
        print('k={}, name = {}, shape = {}'.format(k, v.name, v.get_shape()))
    
    print("Parameters")
    for v in slim.get_model_variables():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))  
    
    
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        
        for i in range(10):
            start_time = time.time()
            pred = sess.run(logits)
            duration = time.time() - start_time
            print ('%s: step %d, duration = %.3f' %(datetime.now(), i, duration))



    
