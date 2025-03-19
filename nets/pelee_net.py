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
"""Contains the definition of the Inception V4 architecture.

As described in http://arxiv.org/abs/1602.07261.

  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import pelee_utils

slim = tf.contrib.slim

def stem_block(inputs, scope=None, reuse=None):
  """Builds stem block for pelee network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'stem_block', [inputs], reuse=reuse):
      #inputs = slim.batch_norm(inputs)
      stem1 = slim.conv2d(inputs, 32, [3, 3], stride=2, scope='stem1')
      stem2 = slim.conv2d(stem1, 16, [1, 1], scope='stem2a')
      stem2 = slim.conv2d(stem2, 32, [3, 3], stride=2, scope='stem2b')
      stem1 = slim.max_pool2d(stem1, [2, 2], stride=2, padding='VALID')
      concate = tf.concat(axis=3, values=[stem1, stem2])
      stem3 = slim.conv2d(concate, 32, [1, 1], scope='stem3')
  return stem3

def dense_block_1(inputs, scope=None, reuse=None):
  """Builds dense block for pelee network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'dense_block_1', [inputs], reuse=reuse):
      results = inputs
      for i in range(3):
        with tf.variable_scope('layer_{}'.format(i + 1)):
          cb1 = slim.conv2d(results, 16, [1, 1], scope='branch1a')
          cb1 = slim.conv2d(cb1, 16, [3, 3], scope='branch1b')
          cb2 = slim.conv2d(results, 16, [1, 1], scope='branch2a')
          cb2 = slim.conv2d(cb2, 16, [3, 3], scope='branch2b')
          cb2 = slim.conv2d(cb2, 16, [3, 3], scope='branch2c')
          results = tf.concat(axis=3, values=[results, cb1, cb2])
  return results

def dense_block_2(inputs, scope=None, reuse=None):
  """Builds dense block for pelee network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'dense_block_2', [inputs], reuse=reuse):
      results = inputs
      for i in range(4):
        with tf.variable_scope('layer_{}'.format(i + 1)):
          cb1 = slim.conv2d(results, 32, [1, 1], scope='branch1a')
          cb1 = slim.conv2d(cb1, 16, [3, 3], scope='branch1b')
          cb2 = slim.conv2d(results, 32, [1, 1], scope='branch2a')
          cb2 = slim.conv2d(cb2, 16, [3, 3], scope='branch2b')
          cb2 = slim.conv2d(cb2, 16, [3, 3], scope='branch2c')
          results = tf.concat(axis=3, values=[results, cb1, cb2])
  return results

def dense_block_3(inputs, scope=None, reuse=None):
  """Builds dense block for pelee network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'dense_block_3', [inputs], reuse=reuse):
      results = inputs
      for i in range(8):
        with tf.variable_scope('layer_{}'.format(i + 1)):
          cb1 = slim.conv2d(results, 64, [1, 1], scope='branch1a')
          cb1 = slim.conv2d(cb1, 16, [3, 3], scope='branch1b')
          cb2 = slim.conv2d(results, 64, [1, 1], scope='branch2a')
          cb2 = slim.conv2d(cb2, 16, [3, 3], scope='branch2b')
          cb2 = slim.conv2d(cb2, 16, [3, 3], scope='branch2c')
          results = tf.concat(axis=3, values=[results, cb1, cb2])
  return results

def dense_block_4(inputs, scope=None, reuse=None):
  """Builds dense block for pelee network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'dense_block_4', [inputs], reuse=reuse):
      results = inputs
      for i in range(6):
        with tf.variable_scope('layer_{}'.format(i + 1)):
          cb1 = slim.conv2d(results, 64, [1, 1], scope='branch1a')
          cb1 = slim.conv2d(cb1, 16, [3, 3], scope='branch1b')
          cb2 = slim.conv2d(results, 64, [1, 1], scope='branch2a')
          cb2 = slim.conv2d(cb2, 16, [3, 3], scope='branch2b')
          cb2 = slim.conv2d(cb2, 16, [3, 3], scope='branch2c')
          results = tf.concat(axis=3, values=[results, cb1, cb2])
  return results

def transition_block_1(inputs, scope=None, reuse=None):
  """Builds transition block for pelee network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'transition_block_1', [inputs], reuse=reuse):
      conv = slim.conv2d(inputs, 128, [1, 1])
      pool = slim.avg_pool2d(conv, [2, 2], stride=2, padding='VALID')
  return pool

def transition_block_2(inputs, scope=None, reuse=None):
  """Builds transition block for pelee network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'transition_block_2', [inputs], reuse=reuse):
      conv = slim.conv2d(inputs, 256, [1, 1])
      pool = slim.avg_pool2d(conv, [2, 2], stride=2, padding='VALID')
  return pool

def transition_block_3(inputs, scope=None, reuse=None):
  """Builds transition block for pelee network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'transition_block_3', [inputs], reuse=reuse):
      conv = slim.conv2d(inputs, 512, [1, 1])
      pool = slim.avg_pool2d(conv, [2, 2], stride=2, padding='VALID')
  return pool

def transition_block_4(inputs, scope=None, reuse=None):
  """Builds transition block for pelee network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'transition_block_4', [inputs], reuse=reuse):
      conv = slim.conv2d(inputs, 704, [1, 1])
  return conv

def attention_block_1(inputs, scope=None, reuse=None):
  """Builds attention block for pelee network."""
  # By default use stride=1 and VALID padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='VALID'):
    with tf.variable_scope(scope, 'attention_block_1', [inputs], reuse=reuse):
      depth = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True, name='spatial_pool')
      depth_sqz = slim.conv2d(depth, 14, [1, 1], scope='depth_squeeze')
      depth_exc = slim.conv2d(depth_sqz, 128, [1, 1], scope='depth_excitation',
                              activation_fn=None)

      space = tf.reduce_mean(inputs, axis=3, keepdims=True, name='depth_pool')
      space = tf.reshape(space, [-1, 1, 1, 28*28])
      space_sqz = slim.conv2d(space, 49, [1, 1], scope="space_squeeze")
      space_exc = slim.conv2d(space_sqz, 28*28, [1, 1], scope="space_excitation",
                              activation_fn=None)
      space_exc = tf.reshape(space_exc, [-1, 28, 28, 1])
      outputs = tf.multiply(tf.multiply(inputs, space_exc), depth_exc) + inputs
  return outputs

def attention_block_2(inputs, scope=None, reuse=None):
  """Builds attention block for pelee network."""
  # By default use stride=1 and VALID padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='VALID'):
    with tf.variable_scope(scope, 'attention_block_2', [inputs], reuse=reuse):
      depth = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True, name='spatial_pool')
      depth_sqz = slim.conv2d(depth, 16, [1, 1], scope='depth_squeeze')
      depth_exc = slim.conv2d(depth_sqz, 256, [1, 1], scope='depth_excitation',
                              activation_fn=None)

      space = tf.reduce_mean(inputs, axis=3, keepdims=True, name='depth_pool')
      space = tf.reshape(space, [-1, 1, 1, 14*14])
      space_sqz = slim.conv2d(space, 14, [1, 1], scope="space_squeeze")
      space_exc = slim.conv2d(space_sqz, 14*14, [1, 1], scope="space_excitation",
                              activation_fn=None)
      space_exc = tf.reshape(space_exc, [-1, 14, 14, 1])
      outputs = tf.multiply(tf.multiply(inputs, space_exc), depth_exc) + inputs
  return outputs

def attention_block_3(inputs, scope=None, reuse=None):
  """Builds attention block for pelee network."""
  # By default use stride=1 and VALID padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='VALID'):
    with tf.variable_scope(scope, 'attention_block_3', [inputs], reuse=reuse):
      depth = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True, name='spatial_pool')
      depth_sqz = slim.conv2d(depth, 32, [1, 1], scope='depth_squeeze')
      depth_exc = slim.conv2d(depth_sqz, 512, [1, 1], scope='depth_excitation',
                              activation_fn=None)

      space = tf.reduce_mean(inputs, axis=3, keepdims=True, name='depth_pool')
      space = tf.reshape(space, [-1, 1, 1, 7*7])
      space_sqz = slim.conv2d(space, 7, [1, 1], scope="space_squeeze")
      space_exc = slim.conv2d(space_sqz, 7*7, [1, 1], scope="space_excitation",
                              activation_fn=None)
      space_exc = tf.reshape(space_exc, [-1, 7, 7, 1])
      outputs = tf.multiply(tf.multiply(inputs, space_exc), depth_exc) + inputs
  return outputs

def attention_block_4(inputs, scope=None, reuse=None):
  """Builds attention block for pelee network."""
  # By default use stride=1 and VALID padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='VALID'):
    with tf.variable_scope(scope, 'attention_block_4', [inputs], reuse=reuse):
      depth = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True, name='spatial_pool')
      depth_sqz = slim.conv2d(depth, 44, [1, 1], scope='depth_squeeze')
      depth_exc = slim.conv2d(depth_sqz, 704, [1, 1], scope='depth_excitation',
                              activation_fn=None)

      space = tf.reduce_mean(inputs, axis=3, keepdims=True, name='depth_pool')
      space = tf.reshape(space, [-1, 1, 1, 7*7])
      space_sqz = slim.conv2d(space, 7, [1, 1], scope="space_squeeze")
      space_exc = slim.conv2d(space_sqz, 7*7, [1, 1], scope="space_excitation",
                              activation_fn=None)
      space_exc = tf.reshape(space_exc, [-1, 7, 7, 1])
      outputs = tf.multiply(tf.multiply(inputs, space_exc), depth_exc) + inputs
  return outputs

def pelee_net_base(inputs, final_endpoint='dense4', scope=None, use_attention=False):
  """Creates the pelee network up to the given final endpoint.

  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
    final_endpoint: specifies the endpoint to construct the network up to.
      It can be one of [ 'stem', 'dense1', 'dense2', 'dense3', 'dense4']
    scope: Optional variable_scope.

  Returns:
    logits: the logits outputs of the model.
    end_points: the set of end_points from the inception model.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
  """
  end_points = {}

  def add_and_check_final(name, net):
    end_points[name] = net
    return name == final_endpoint

  with tf.variable_scope(scope, 'pelee_net', [inputs]):
    net = stem_block(inputs)
    if add_and_check_final('stem', net): return net, end_points
    net = transition_block_1(dense_block_1(net))
    if use_attention: net = attention_block_1(net)
    if add_and_check_final('dense1', net): return net, end_points
    net = transition_block_2(dense_block_2(net))
    if use_attention: net = attention_block_2(net)
    if add_and_check_final('dense2', net): return net, end_points
    net = transition_block_3(dense_block_3(net))
    if use_attention: net = attention_block_3(net)
    if add_and_check_final('dense3', net): return net, end_points
    net = transition_block_4(dense_block_4(net))
    if use_attention: net = attention_block_4(net)
    if add_and_check_final('dense4', net): return net, end_points
  raise ValueError('Unknown final endpoint %s' % final_endpoint)


def pelee_net(inputs, num_classes=1001, is_training=True,
              dropout_keep_prob=0.95,
              reuse=None,
              scope='pelee_net',
              create_aux_logits=False,
              use_attention=False):
  """Creates the pelee net model.

  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before dropout)
      are returned instead.
    is_training: whether is training or not.
    dropout_keep_prob: float, the fraction to keep before final layer.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    create_aux_logits: Whether to include the auxiliary logits.

  Returns:
    net: a Tensor with the logits (pre-softmax activations) if num_classes
      is a non-zero integer, or the non-dropped input to the logits layer
      if num_classes is 0 or None.
    end_points: the set of end_points from the inception model.
  """
  end_points = {}
  with tf.variable_scope(scope, 'pelee_net', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = pelee_net_base(inputs, scope=scope, use_attention=use_attention)

      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                          stride=1, padding='SAME'):
        # Auxiliary Head logits
        if create_aux_logits and num_classes:
          with tf.variable_scope('AuxLogits'):
            # 14 x 14 x 256
            aux_logits = end_points['dense2']
            # 4 x 4 x 256
            aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3,
                                         padding='VALID',
                                         scope='AvgPool_1a_5x5')
            # 4 x 4 x 128
            aux_logits = slim.conv2d(aux_logits, 128, [1, 1],
                                     scope='Conv2d_1b_1x1')
            # 1 x 1 x 768
            aux_logits = slim.conv2d(aux_logits, 768,
                                     aux_logits.get_shape()[1:3],
                                     padding='VALID', scope='Conv2d_2a')
            aux_logits = slim.flatten(aux_logits)
            aux_logits = slim.fully_connected(aux_logits, num_classes,
                                              activation_fn=None,
                                              scope='Aux_logits')
            end_points['AuxLogits'] = aux_logits

        # Final pooling and prediction
        # TODO(sguada,arnoegw): Consider adding a parameter global_pool which
        # can be set to False to disable pooling here (as in resnet_*()).
        with tf.variable_scope('Logits'):
          # 7 x 7 x 704
          kernel_size = net.get_shape()[1:3]
          if kernel_size.is_fully_defined():
            net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                  scope='AvgPool_1a')
          else:
            net = tf.reduce_mean(net, [1, 2], keep_dims=True,
                                 name='global_pool')
          end_points['global_pool'] = net
          if not num_classes:
            return net, end_points
          # 1 x 1 x 1536
          net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b')
          net = slim.flatten(net, scope='PreLogitsFlatten')
          end_points['PreLogitsFlatten'] = net
          # 1536
          logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                        normalizer_fn=None,
                                        scope='Logits')
          end_points['Logits'] = logits
          end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
    return logits, end_points
pelee_net.default_image_size = 224


def pelee_net_multilabel(inputs, num_classes=1001, is_training=True,
              dropout_keep_prob=0.95,
              reuse=None,
              scope='pelee_net',
              create_aux_logits=False,
              use_attention=False):
  """Creates the pelee net model.

  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before dropout)
      are returned instead.
    is_training: whether is training or not.
    dropout_keep_prob: float, the fraction to keep before final layer.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    create_aux_logits: Whether to include the auxiliary logits.

  Returns:
    net: a Tensor with the logits (pre-softmax activations) if num_classes
      is a non-zero integer, or the non-dropped input to the logits layer
      if num_classes is 0 or None.
    end_points: the set of end_points from the inception model.
  """
  end_points = {}
  with tf.variable_scope(scope, 'pelee_net', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = pelee_net_base(inputs, scope=scope, use_attention=use_attention)

      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                          stride=1, padding='SAME'):
        # Auxiliary Head logits
        if create_aux_logits and num_classes:
          with tf.variable_scope('AuxLogits'):
            # 14 x 14 x 256
            aux_logits = end_points['dense2']
            # 4 x 4 x 256
            aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3,
                                         padding='VALID',
                                         scope='AvgPool_1a_5x5')
            # 4 x 4 x 128
            aux_logits = slim.conv2d(aux_logits, 128, [1, 1],
                                     scope='Conv2d_1b_1x1')
            # 1 x 1 x 768
            aux_logits = slim.conv2d(aux_logits, 768,
                                     aux_logits.get_shape()[1:3],
                                     padding='VALID', scope='Conv2d_2a')
            aux_logits = slim.flatten(aux_logits)
            aux_logits = slim.fully_connected(aux_logits, num_classes,
                                              activation_fn=None,
                                              scope='Aux_logits')
            end_points['AuxLogits'] = aux_logits

        # Final pooling and prediction
        # TODO(sguada,arnoegw): Consider adding a parameter global_pool which
        # can be set to False to disable pooling here (as in resnet_*()).
        with tf.variable_scope('Logits'):
          # 7 x 7 x 704
          kernel_size = net.get_shape()[1:3]
          if kernel_size.is_fully_defined():
            net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                  scope='AvgPool_1a')
          else:
            net = tf.reduce_mean(net, [1, 2], keep_dims=True,
                                 name='global_pool')
          end_points['global_pool'] = net
          if not num_classes:
            return net, end_points
          # 1 x 1 x 1536
          net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b')
          net = slim.flatten(net, scope='PreLogitsFlatten')
          end_points['PreLogitsFlatten'] = net
          # 1536
          logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                        normalizer_fn=None,
                                        scope='Logits')
          end_points['Logits'] = logits
          end_points['Predictions'] = tf.nn.sigmoid(logits, name='Predictions')
    return logits, end_points
pelee_net_multilabel.default_image_size = 224


pelee_arg_scope = pelee_utils.pelee_arg_scope
