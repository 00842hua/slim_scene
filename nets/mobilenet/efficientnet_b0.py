
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools

import tensorflow as tf

from nets.mobilenet import conv_blocks as ops
from nets.mobilenet import mobilenet as lib

slim = tf.contrib.slim
op = lib.op

expand_input = ops.expand_input_by_factor
def my_swish(x):
  return tf.nn.relu6(x)

# pyformat: disable
# Architecture: https://arxiv.org/abs/1801.04381
V2_DEF = dict(
    defaults={
        # Note: these parameters of batch norm affect the architecture
        # that's why they are here and not in training_scope.
        (slim.batch_norm,): {'center': True, 'scale': True},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            'normalizer_fn': slim.batch_norm
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
        op(slim.conv2d, stride=2,num_outputs=32,activation_fn=my_swish,kernel_size=[3,3]),

        op(ops.expanded_conv, stride=1, num_outputs=16,kernel_size=[3,3],expansion_size=expand_input(1,divisible_by=1),activation_fn=my_swish,se=True),

        op(ops.expanded_conv, stride=2, num_outputs=24,kernel_size=[3,3],expansion_size=expand_input(6,divisible_by=1),activation_fn=my_swish,se=True),
        op(ops.expanded_conv, stride=1, num_outputs=24,kernel_size=[3,3],expansion_size=expand_input(6,divisible_by=1),activation_fn=my_swish,se=True),

        op(ops.expanded_conv, stride=2, num_outputs=40,kernel_size=[5,5],expansion_size=expand_input(6,divisible_by=1),activation_fn=my_swish,se=True),
        op(ops.expanded_conv, stride=1, num_outputs=40,kernel_size=[5,5],expansion_size=expand_input(6,divisible_by=1),activation_fn=my_swish,se=True),

        op(ops.expanded_conv, stride=2, num_outputs=80,kernel_size=[3,3],expansion_size=expand_input(6,divisible_by=1),activation_fn=my_swish,se=True),
        op(ops.expanded_conv, stride=1, num_outputs=80,kernel_size=[3,3],expansion_size=expand_input(6,divisible_by=1),activation_fn=my_swish,se=True),
        op(ops.expanded_conv, stride=1, num_outputs=80,kernel_size=[3,3],expansion_size=expand_input(6,divisible_by=1),activation_fn=my_swish,se=True),

        op(ops.expanded_conv, stride=1, num_outputs=112,kernel_size=[5,5],expansion_size=expand_input(6,divisible_by=1),activation_fn=my_swish,se=True),
        op(ops.expanded_conv, stride=1, num_outputs=112,kernel_size=[5,5],expansion_size=expand_input(6,divisible_by=1),activation_fn=my_swish,se=True),
        op(ops.expanded_conv, stride=1, num_outputs=112,kernel_size=[5,5],expansion_size=expand_input(6,divisible_by=1),activation_fn=my_swish,se=True),

        op(ops.expanded_conv, stride=2, num_outputs=192,kernel_size=[5,5],expansion_size=expand_input(6,divisible_by=1),activation_fn=my_swish,se=True),
        op(ops.expanded_conv, stride=1, num_outputs=192,kernel_size=[5,5],expansion_size=expand_input(6,divisible_by=1),activation_fn=my_swish,se=True),
        op(ops.expanded_conv, stride=1, num_outputs=192,kernel_size=[5,5],expansion_size=expand_input(6,divisible_by=1),activation_fn=my_swish,se=True),
        op(ops.expanded_conv, stride=1, num_outputs=192,kernel_size=[5,5],expansion_size=expand_input(6,divisible_by=1),activation_fn=my_swish,se=True),

        op(ops.expanded_conv, stride=1, num_outputs=320,kernel_size=[3,3],expansion_size=expand_input(6,divisible_by=1),activation_fn=my_swish,se=True),

        op(slim.conv2d, stride=1,num_outputs=1280,activation_fn=my_swish,kernel_size=1),
    ],
)
# pyformat: enable


@slim.add_arg_scope
def mobilenet(input_tensor,
              num_classes=1001,
              depth_multiplier=1.0,
              scope='efficientnet_b0',
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
    channels in each layer.
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


def wrapped_partial(func, *args, **kwargs):
  partial_func = functools.partial(func, *args, **kwargs)
  functools.update_wrapper(partial_func, func)
  return partial_func


# Wrappers for mobilenet v2 with depth-multipliers. Be noticed that
# 'finegrain_classification_mode' is set to True, which means the embedding
# layer will not be shrinked when given a depth-multiplier < 1.0.
# mobilenet_v2_140 = wrapped_partial(mobilenet, depth_multiplier=1.4)
# mobilenet_v2_050 = wrapped_partial(mobilenet, depth_multiplier=0.50,
#                                   finegrain_classification_mode=True)
# mobilenet_v2_035 = wrapped_partial(mobilenet, depth_multiplier=0.35,
#                                   finegrain_classification_mode=True)


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
  return lib.training_scope(**kwargs)


__all__ = ['training_scope', 'mobilenet_base', 'mobilenet', 'V2_DEF']


def model_size():
  params = tf.trainable_variables()
  size = 0
  for x in params:
    sz = 1
    for dim in x.get_shape():
      sz *= dim.value
    size += sz
  return size


if __name__ == '__main__':
  images = tf.placeholder(tf.float32,[None,256,256,3],name='images')

  with tf.Session() as sess:
    logits,end_points = mobilenet(images)
    print(logits.shape,end_points['Predictions'].shape)
    print('Size:',model_size())
    