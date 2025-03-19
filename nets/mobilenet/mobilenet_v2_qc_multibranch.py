# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of Mobilenet V2.

Architecture: https://arxiv.org/abs/1801.04381

The base model gives 72.2% accuracy on ImageNet, with 300MMadds,
3.4 M parameters.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools

import tensorflow as tf

from nets.mobilenet import conv_blocks_qc as ops
from nets.mobilenet import mobilenet as lib

slim = tf.contrib.slim
op = lib.op

expand_input = ops.expand_input_by_factor

# pyformat: disable
# Architecture: https://arxiv.org/abs/1801.04381
V2_DEF = dict(
    defaults={
        # Note: these parameters of batch norm affect the architecture
        # that's why they are here and not in training_scope.
        (slim.batch_norm,): {'center': True, 'scale': True},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            # 'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6
            'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu
        },
        (ops.expanded_conv,): {
            'expansion_size': expand_input(6),
            'split_expansion': 1,
            'normalizer_fn': slim.batch_norm,
            'residual': True
        },
        (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
    },
    spec=[
        op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
        op(slim.conv2d, stride=1, num_outputs=32, kernel_size=[1, 1]),
        op(ops.expanded_conv,
           expansion_size=expand_input(1, divisible_by=1),
           num_outputs=16),
        op(ops.expanded_conv, stride=2, num_outputs=24),
        op(ops.expanded_conv, stride=1, num_outputs=24),
        op(ops.expanded_conv, stride=2, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=2, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=96),
        op(ops.expanded_conv, stride=1, num_outputs=96),
        op(ops.expanded_conv, stride=1, num_outputs=96),
        op(ops.expanded_conv, stride=2, num_outputs=160),
        op(ops.expanded_conv, stride=1, num_outputs=160),
        op(ops.expanded_conv, stride=1, num_outputs=160),
        op(ops.expanded_conv, stride=1, num_outputs=320),
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
    ],
)

V2_BRANCH_DEF = dict(
    defaults={
        # Note: these parameters of batch norm affect the architecture
        # that's why they are here and not in training_scope.
        (slim.batch_norm,): {'center': True, 'scale': True},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            # 'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6
            'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu
        },
        (ops.expanded_conv,): {
            'expansion_size': expand_input(6),
            'split_expansion': 1,
            'normalizer_fn': slim.batch_norm,
            'residual': True
        },
        (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
    },
    spec=[
        op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),  # layer_1  224*224-->112*112
        op(slim.conv2d, stride=1, num_outputs=32, kernel_size=[1, 1]),  # layer_2
        op(ops.expanded_conv,
           expansion_size=expand_input(1, divisible_by=1),
           num_outputs=16),  # layer_3
        op(ops.expanded_conv, stride=2, num_outputs=24),  # layer_4  112*112-->56*56
        op(ops.expanded_conv, stride=1, num_outputs=24),  # layer_5
        op(ops.expanded_conv, stride=2, num_outputs=32),  # layer_6  56*56-->28*28
        op(ops.expanded_conv, stride=1, num_outputs=32),  # layer_7
        op(ops.expanded_conv, stride=1, num_outputs=32),  # layer_8
        op(ops.expanded_conv, stride=2, num_outputs=64),  # layer_9  28*28-->14*14
        op(ops.expanded_conv, stride=1, num_outputs=64),  # layer_10
        op(ops.expanded_conv, stride=1, num_outputs=64),  # layer_11
        op(ops.expanded_conv, stride=1, num_outputs=64),  # layer_12
        op(ops.expanded_conv, stride=1, num_outputs=96),  # layer_13
        op(ops.expanded_conv, stride=1, num_outputs=96),  # layer_14
        op(ops.expanded_conv, stride=1, num_outputs=96),  # layer_15
        op(ops.expanded_conv, stride=2, num_outputs=160),  # layer_16  14*14-->7*7
        op(ops.expanded_conv, stride=1, num_outputs=160),  # layer_17
        op(ops.expanded_conv, stride=1, num_outputs=160),  # layer_18
        op(ops.expanded_conv, stride=1, num_outputs=320),  # layer_19
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)  # layer_20
    ],
)


# pyformat: enable


@slim.add_arg_scope
def mobilenet_two_branch(input_tensor,
                         num_classes=1001,
                         num_classes_task2=0,
                         depth_multiplier=1.0,
                         depth_multiplier_branch2=1.0,  # 分支2的depth multiplier，以支持更小的分支二
                         branch2_input=15,  # 分支2从第几个op开始
                         scope='MobilenetV2',
                         conv_defs=None,
                         finegrain_classification_mode=False,
                         min_depth=None,
                         divisible_by=None,
                         activation_fn=None,
                         **kwargs):
    is_training = kwargs.get('is_training', False)
    _, endpoints = mobilenet(input_tensor, num_classes=num_classes, depth_multiplier=depth_multiplier,
                             scope=scope, conv_defs=conv_defs,
                             finegrain_classification_mode=finegrain_classification_mode,
                             min_depth=min_depth, divisible_by=divisible_by, activation_fn=activation_fn,
                             **kwargs)

    branch_args = copy.deepcopy(kwargs)
    branch_args['conv_defs'] = V2_BRANCH_DEF
    branch_args['conv_defs']['spec'] = branch_args['conv_defs']['spec'][branch2_input:]
    branch_args['multiplier'] = depth_multiplier_branch2
    # 保持最后一层的输出数量仍为1280，即对应独立mob模型的finegrain_classification_mode=True
    if depth_multiplier_branch2 < 1 and len(branch_args['conv_defs']['spec']) > 0:
        branch_args['conv_defs']['spec'][-1].params['num_outputs'] /= depth_multiplier_branch2
    input_layer = 'layer_{}'.format(branch2_input)
    print("Branch2 input layer is {}".format(input_layer))
    print("Branch2 op params as as follows: ")
    for bop in branch_args['conv_defs']['spec']:
        print(bop.params)

    with tf.variable_scope("{}_branch2".format(scope)) as scope:
        inputs = tf.identity(endpoints[input_layer], name='input')
        batch_norm_params = {
            'decay': 0.997,
            'is_training': is_training,
            'trainable': is_training,
        }
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):  # 此处需打开bn的更新
            net, branch_end_points = lib.mobilenet_base(inputs, scope=scope, **branch_args)
        net = tf.identity(net, name='embedding')

        with tf.variable_scope('Logits'):
            net = lib.global_pool(net)
            branch_end_points['global_pool'] = net
            net = slim.dropout(net, scope='Dropout', is_training=is_training)
            # 1 x 1 x num_classes
            # Note: legacy scope name.
            logits = slim.conv2d(
                net,
                num_classes_task2, [1, 1],
                activation_fn=None,
                normalizer_fn=None,
                biases_initializer=tf.zeros_initializer(),
                scope='Conv2d_1c_1x1')

            logits = tf.squeeze(logits, [1, 2])

            logits = tf.identity(logits, name='output')
        branch_end_points['Logits'] = logits
        branch_end_points['Predictions_softmax'] = tf.nn.softmax(logits, name='Predictions_softmax')
        branch_end_points['Predictions_sigmoid'] = tf.nn.sigmoid(logits, name='Predictions_sigmoid')

    endpoints['Logits_task2'] = logits
    endpoints['Predictions_softmax_concat'] = tf.concat([endpoints['Predictions_softmax'],
                                                         branch_end_points['Predictions_softmax']],
                                                        axis=-1,
                                                        name='Predictions_softmax_concat')
    endpoints['Predictions_sigmoid_concat'] = tf.concat([endpoints['Predictions_sigmoid'],
                                                         branch_end_points['Predictions_sigmoid']],
                                                        axis=-1,
                                                        name='Predictions_sigmoid_concat')

    for k, v in branch_end_points.items():
        endpoints['branch2/{}'.format(k)] = v

    return logits, endpoints


mobilenet_two_branch.default_image_size = 224


@slim.add_arg_scope
def mobilenet_two_branch_heatmap(input_tensor,
                                 num_classes=1001,
                                 num_classes_task2=0,
                                 depth_multiplier=1.0,
                                 depth_multiplier_branch2=1.0,  # 分支2的depth multiplier，以支持更小的分支二
                                 branch2_input=15,  # 分支2从第几个op开始
                                 scope='MobilenetV2',
                                 conv_defs=None,
                                 finegrain_classification_mode=False,
                                 min_depth=None,
                                 divisible_by=None,
                                 activation_fn=None,
                                 heatmap_class_index=None,    # 提取heatmap的类别index
                                 **kwargs):
    assert isinstance(heatmap_class_index, int), "heatmap_class_index must be an integer"
    assert (0 < heatmap_class_index < num_classes), \
        "heatmap_class_index must be in [0, {})".format(num_classes)

    is_training = kwargs.get('is_training', False)
    _, endpoints = mobilenet(input_tensor, num_classes=num_classes, depth_multiplier=depth_multiplier,
                             scope=scope, conv_defs=conv_defs,
                             finegrain_classification_mode=finegrain_classification_mode,
                             min_depth=min_depth, divisible_by=divisible_by, activation_fn=activation_fn,
                             **kwargs)

    branch_args = copy.deepcopy(kwargs)
    branch_args['conv_defs'] = V2_BRANCH_DEF
    branch_args['conv_defs']['spec'] = branch_args['conv_defs']['spec'][branch2_input:]
    branch_args['multiplier'] = depth_multiplier_branch2
    # 保持最后一层的输出数量仍为1280，即对应独立mob模型的finegrain_classification_mode=True
    if depth_multiplier_branch2 < 1 and len(branch_args['conv_defs']['spec']) > 0:
        branch_args['conv_defs']['spec'][-1].params['num_outputs'] /= depth_multiplier_branch2
    input_layer = 'layer_{}'.format(branch2_input)
    print("Branch2 input layer is {}".format(input_layer))
    print("Branch2 op params as as follows: ")
    for bop in branch_args['conv_defs']['spec']:
        print(bop.params)

    with tf.variable_scope("{}_branch2".format(scope)) as scope2:
        inputs = tf.identity(endpoints[input_layer], name='input')

        batch_norm_params = {
            'decay': 0.997,
            'is_training': is_training,
            'trainable': is_training,
            'center': True,
            'scale': True
        }
        other_norm_params = {  # params for layer_norm and layer_norm
            'trainable': is_training,
            'center': True,
            'scale': True
        }

        # add heatmap attention, trainable=False
        # 此处需打开bn, ln or in的更新. 由于通道数只有一个，ln或in的实际表现应该一致
        with slim.arg_scope([slim.batch_norm], **batch_norm_params), \
             slim.arg_scope([slim.layer_norm], **other_norm_params), \
             slim.arg_scope([slim.instance_norm], **other_norm_params):
            cam_conv = endpoints['layer_20']
            heatmap_att = slim.conv2d(cam_conv, num_outputs=1, kernel_size=[1, 1], stride=1, padding='SAME',
                                      # activation_fn=None, normalizer_fn=None, trainable=False, scope="heatmap_att")
                                      # activation_fn=tf.nn.relu, normalizer_fn=None, trainable=False, scope="heatmap_att")
                                      # activation_fn=None, normalizer_fn=slim.batch_norm, trainable=False, scope="heatmap_att")
                                      # activation_fn=None, normalizer_fn=slim.layer_norm, trainable=False, scope="heatmap_att")
                                      activation_fn=None, normalizer_fn=slim.instance_norm, trainable=False, scope="heatmap_att")
        # 若w, h不等，需要resize一下。
        if inputs.shape[1:3] != heatmap_att.shape[1:3]:
            heatmap_att_reshape = tf.image.resize_bilinear(heatmap_att, inputs.shape[1:3])
            heatmap_att = heatmap_att_reshape
        tf.summary.image('input', input_tensor, max_outputs=4)
        tf.summary.image('heatmap', heatmap_att, max_outputs=4)
        tf.summary.histogram("att_cam_conv", cam_conv)
        tf.summary.histogram("att_inputs", inputs)
        tf.summary.histogram("att_heatmap", heatmap_att)

        inputs = tf.multiply(inputs, heatmap_att, name="att_mul")
        # inputs = tf.concat([inputs, heatmap_att], axis=-1, name="att_concat")
        # inputs = tf.add(inputs, heatmap_att, name="att_add")
        # inputs = tf.add(inputs, tf.multiply(inputs, heatmap_att, name="att_mul"), name="att_mul_add")
        # inputs = tf.concat([inputs, tf.multiply(inputs, heatmap_att, name="att_mul")], axis=-1, name="att_mul_concat")
        tf.summary.histogram("att_outputs", inputs)

        with slim.arg_scope([slim.batch_norm], **batch_norm_params):  # 此处需打开bn的更新
            net, branch_end_points = lib.mobilenet_base(inputs, scope=scope2, **branch_args)
        net = tf.identity(net, name='embedding')

        with tf.variable_scope('Logits'):
            net = lib.global_pool(net)
            branch_end_points['global_pool'] = net
            net = slim.dropout(net, scope='Dropout', is_training=is_training)
            # 1 x 1 x num_classes
            # Note: legacy scope name.
            logits = slim.conv2d(
                net,
                num_classes_task2, [1, 1],
                activation_fn=None,
                normalizer_fn=None,
                biases_initializer=tf.zeros_initializer(),
                scope='Conv2d_1c_1x1')

            logits = tf.squeeze(logits, [1, 2])

            logits = tf.identity(logits, name='output')
        branch_end_points['Logits'] = logits
        branch_end_points['Predictions_softmax'] = tf.nn.softmax(logits, name='Predictions_softmax')
        branch_end_points['Predictions_sigmoid'] = tf.nn.sigmoid(logits, name='Predictions_sigmoid')

    endpoints['Logits_task2'] = logits
    endpoints['Predictions_softmax_concat'] = tf.concat([endpoints['Predictions_softmax'],
                                                         branch_end_points['Predictions_softmax']],
                                                        axis=-1,
                                                        name='Predictions_softmax_concat')
    endpoints['Predictions_sigmoid_concat'] = tf.concat([endpoints['Predictions_sigmoid'],
                                                         branch_end_points['Predictions_sigmoid']],
                                                        axis=-1,
                                                        name='Predictions_sigmoid_concat')

    for k, v in branch_end_points.items():
        endpoints['branch2/{}'.format(k)] = v

    return logits, endpoints

mobilenet_two_branch_heatmap.default_image_size = 224


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


# Wrappers for mobilenet v2 with depth-multipliers. Be noticed that
# 'finegrain_classification_mode' is set to True, which means the embedding
# layer will not be shrinked when given a depth-multiplier < 1.0.
mobilenet_two_branch_140 = wrapped_partial(mobilenet_two_branch, depth_multiplier_branch2=1.4)
mobilenet_two_branch_050 = wrapped_partial(mobilenet_two_branch, depth_multiplier_branch2=0.50)
mobilenet_two_branch_035 = wrapped_partial(mobilenet_two_branch, depth_multiplier_branch2=0.35)
mobilenet_two_branch_050_input16 = wrapped_partial(mobilenet_two_branch, depth_multiplier_branch2=0.35,
                                                   branch2_input=16)
mobilenet_two_branch_050_input17 = wrapped_partial(mobilenet_two_branch, depth_multiplier_branch2=0.35,
                                                   branch2_input=17)
mobilenet_two_branch_050_input18 = wrapped_partial(mobilenet_two_branch, depth_multiplier_branch2=0.35,
                                                   branch2_input=18)
mobilenet_two_branch_050_input19 = wrapped_partial(mobilenet_two_branch, depth_multiplier_branch2=0.35,
                                                   branch2_input=19)
mobilenet_two_branch_050_input20 = wrapped_partial(mobilenet_two_branch, depth_multiplier_branch2=0.35,
                                                   branch2_input=20)
mobilenet_two_branch_035_input16 = wrapped_partial(mobilenet_two_branch, depth_multiplier_branch2=0.35,
                                                   branch2_input=16)
mobilenet_two_branch_035_input17 = wrapped_partial(mobilenet_two_branch, depth_multiplier_branch2=0.35,
                                                   branch2_input=17)
mobilenet_two_branch_035_input18 = wrapped_partial(mobilenet_two_branch, depth_multiplier_branch2=0.35,
                                                   branch2_input=18)
mobilenet_two_branch_035_input9 = wrapped_partial(mobilenet_two_branch, depth_multiplier_branch2=0.35, branch2_input=9)

mobilenet_two_branch_heatmap_035 = wrapped_partial(mobilenet_two_branch_heatmap, depth_multiplier_branch2=0.35)


@slim.add_arg_scope
def mobilenet(input_tensor,
              num_classes=1001,
              depth_multiplier=1.0,
              scope='MobilenetV2',
              conv_defs=None,
              finegrain_classification_mode=False,
              min_depth=None,
              divisible_by=None,
              activation_fn=None,
              **kwargs):
    """Creates mobilenet V2 network.

    Inference mode is created by default. To create training use training_scope
    below.

    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
       logits, endpoints = mobilenet_v2.mobilenet(input_tensor)

    Args:
      input_tensor: The input tensor
      num_classes: number of classes
      depth_multiplier: The multiplier applied to scale number of
      channels in each layer. Note: this is called depth multiplier in the
      paper but the name is kept for consistency with slim's model builder.
      scope: Scope of the operator
      conv_defs: Allows to override default conv def.
      finegrain_classification_mode: When set to True, the model
      will keep the last layer large even for small multipliers. Following
      https://arxiv.org/abs/1801.04381
      suggests that it improves performance for ImageNet-type of problems.
        *Note* ignored if final_endpoint makes the builder exit earlier.
      min_depth: If provided, will ensure that all layers will have that
      many channels after application of depth multiplier.
      divisible_by: If provided will ensure that all layers # channels
      will be divisible by this number.
      activation_fn: Activation function to use, defaults to tf.nn.relu6 if not
        specified.
      **kwargs: passed directly to mobilenet.mobilenet:
        prediction_fn- what prediction function to use.
        reuse-: whether to reuse variables (if reuse set to true, scope
        must be given).
    Returns:
      logits/endpoints pair

    Raises:
      ValueError: On invalid arguments
    """
    if conv_defs is None:
        conv_defs = V2_DEF
    if 'multiplier' in kwargs:
        raise ValueError('mobilenetv2 doesn\'t support generic '
                         'multiplier parameter use "depth_multiplier" instead.')
    if finegrain_classification_mode:
        conv_defs = copy.deepcopy(conv_defs)
        if depth_multiplier < 1:
            conv_defs['spec'][-1].params['num_outputs'] /= depth_multiplier
    if activation_fn:
        conv_defs = copy.deepcopy(conv_defs)
        defaults = conv_defs['defaults']
        conv_defaults = (
            defaults[(slim.conv2d, slim.fully_connected, slim.separable_conv2d)])
        conv_defaults['activation_fn'] = activation_fn

    depth_args = {}
    # NB: do not set depth_args unless they are provided to avoid overriding
    # whatever default depth_multiplier might have thanks to arg_scope.
    if min_depth is not None:
        depth_args['min_depth'] = min_depth
    if divisible_by is not None:
        depth_args['divisible_by'] = divisible_by

    with slim.arg_scope((lib.depth_multiplier,), **depth_args):
        return lib.mobilenet(
            input_tensor,
            num_classes=num_classes,
            conv_defs=conv_defs,
            scope=scope,
            multiplier=depth_multiplier,
            **kwargs)


mobilenet.default_image_size = 224


@slim.add_arg_scope
def mobilenet_multilabel(input_tensor,
                         num_classes=1001,
                         depth_multiplier=1.0,
                         scope='MobilenetV2',
                         conv_defs=None,
                         finegrain_classification_mode=False,
                         min_depth=None,
                         divisible_by=None,
                         activation_fn=None,
                         **kwargs):
    if conv_defs is None:
        conv_defs = V2_DEF
    if 'multiplier' in kwargs:
        raise ValueError('mobilenetv2 doesn\'t support generic '
                         'multiplier parameter use "depth_multiplier" instead.')
    if finegrain_classification_mode:
        conv_defs = copy.deepcopy(conv_defs)
        if depth_multiplier < 1:
            conv_defs['spec'][-1].params['num_outputs'] /= depth_multiplier
    if activation_fn:
        conv_defs = copy.deepcopy(conv_defs)
        defaults = conv_defs['defaults']
        conv_defaults = (
            defaults[(slim.conv2d, slim.fully_connected, slim.separable_conv2d)])
        conv_defaults['activation_fn'] = activation_fn

    depth_args = {}
    # NB: do not set depth_args unless they are provided to avoid overriding
    # whatever default depth_multiplier might have thanks to arg_scope.
    if min_depth is not None:
        depth_args['min_depth'] = min_depth
    if divisible_by is not None:
        depth_args['divisible_by'] = divisible_by

    with slim.arg_scope((lib.depth_multiplier,), **depth_args):
        return lib.mobilenet(
            input_tensor,
            num_classes=num_classes,
            conv_defs=conv_defs,
            scope=scope,
            multiplier=depth_multiplier,
            prediction_fn=tf.nn.sigmoid,
            **kwargs)


mobilenet_multilabel.default_image_size = 224

# Wrappers for mobilenet v2 with depth-multipliers. Be noticed that
# 'finegrain_classification_mode' is set to True, which means the embedding
# layer will not be shrinked when given a depth-multiplier < 1.0.
mobilenet_v2_140 = wrapped_partial(mobilenet, depth_multiplier=1.4)
mobilenet_v2_050 = wrapped_partial(mobilenet, depth_multiplier=0.50,
                                   finegrain_classification_mode=True)
mobilenet_v2_035 = wrapped_partial(mobilenet, depth_multiplier=0.35,
                                   finegrain_classification_mode=True)
mobilenet_v2_140_multilabel = wrapped_partial(mobilenet_multilabel, depth_multiplier=1.4)


@slim.add_arg_scope
def mobilenet_base(input_tensor, depth_multiplier=1.0, **kwargs):
    """Creates base of the mobilenet (no pooling and no logits) ."""
    return mobilenet(input_tensor,
                     depth_multiplier=depth_multiplier,
                     base_only=True, **kwargs)


def training_scope(**kwargs):
    """Defines MobilenetV2 training scope.

    Usage:
       with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
         logits, endpoints = mobilenet_v2.mobilenet(input_tensor)

    with slim.

    Args:
      **kwargs: Passed to mobilenet.training_scope. The following parameters
      are supported:
        weight_decay- The weight decay to use for regularizing the model.
        stddev-  Standard deviation for initialization, if negative uses xavier.
        dropout_keep_prob- dropout keep probability
        bn_decay- decay for the batch norm moving averages.

    Returns:
      An `arg_scope` to use for the mobilenet v2 model.
    """
    return lib.training_scope(regularize_depthwise=True, **kwargs)


__all__ = ['training_scope', 'mobilenet_base', 'mobilenet', 'V2_DEF']
