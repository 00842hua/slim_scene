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
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from nets import alexnet
from nets import cifarnet
from nets import inception
from nets import lenet
from nets import mobilenet_v1
from nets import mobilenet_v1_qc
from nets import mobilenet_v1_qc_se
from nets import mobilenet_v1_qc_mish
from nets import mobilenet_v1_multilabel
from nets import overfeat
from nets import resnet_v1
from nets import resnet_v2
from nets import vgg
from nets.mobilenet import mobilenet_v2
from nets.mobilenet import efficientnet_b0
from nets.mobilenet import mobilenet_v3_large
from nets.nasnet import nasnet
from nets.nasnet import pnasnet
from nets import pelee_net
from nets import shufflenet
from nets import shufflenet_multilabel
from nets import shufflenet_v2
from nets import shufflenet_v2_multilabel
from nets import densenet_multilabel
# from nets import mobilenet_v3
from nets import efficientnet_b0_wf
#from nets.EfficientNet import efficientnet_builder
from nets.mnasnet import mnasnet_models
from nets.mnasnet import mnasnet
from nets.fairnas import fairnas_a
from nets.fairnas import fairnas_b
from nets.fairnas import fairnas_c
from nets.moga import moga_a
from nets.moga import moga_b
from nets.moga import moga_c
from nets.mnasnet import mnasnet_fix
from nets.mobilenet import mobilenet_v2_qc
from nets.mobilenet import mobilenet_v2_qc_multibranch
from nets.mobilenet import mobilenet_v3
from nets import VPN_Network

slim = tf.contrib.slim

networks_map = {'alexnet_v2': alexnet.alexnet_v2,
                'cifarnet': cifarnet.cifarnet,
                'overfeat': overfeat.overfeat,
                'vgg_a': vgg.vgg_a,
                'vgg_16': vgg.vgg_16,
                'vgg_19': vgg.vgg_19,
                'inception_v1': inception.inception_v1,
                'inception_v2': inception.inception_v2,
                'inception_v3': inception.inception_v3,
                'inception_v4': inception.inception_v4,
                'inception_resnet_v2': inception.inception_resnet_v2,
                'inception_resnet_v2_center': inception.inception_resnet_v2_center,
                'inception_resnet_v2_dropblock': inception.inception_resnet_v2_dropblock,
                'lenet': lenet.lenet,
                'resnet_v1_50': resnet_v1.resnet_v1_50,
                'resnet_v1_101': resnet_v1.resnet_v1_101,
                'resnet_v1_152': resnet_v1.resnet_v1_152,
                'resnet_v1_200': resnet_v1.resnet_v1_200,
                'resnet_v2_50': resnet_v2.resnet_v2_50,
                'resnet_v2_101': resnet_v2.resnet_v2_101,
                'resnet_v2_152': resnet_v2.resnet_v2_152,
                'resnet_v2_200': resnet_v2.resnet_v2_200,
                'mobilenet_v1': mobilenet_v1.mobilenet_v1,
                'mobilenet_v1_075': mobilenet_v1.mobilenet_v1_075,
                'mobilenet_v1_050': mobilenet_v1.mobilenet_v1_050,
                'mobilenet_v1_025': mobilenet_v1.mobilenet_v1_025,
                'mobilenet_v2': mobilenet_v2.mobilenet,
                'mobilenet_v2_multilabel': mobilenet_v2.mobilenet_multilabel,
                'mobilenet_v2_140': mobilenet_v2.mobilenet_v2_140,
                'mobilenet_v2_140_multilabel': mobilenet_v2.mobilenet_v2_140_multilabel,
                'mobilenet_v2_035': mobilenet_v2.mobilenet_v2_035,
                'mobilenet_v1_qc': mobilenet_v1_qc.mobilenet_v1,
                'mobilenet_v1_qc_mish': mobilenet_v1_qc_mish.mobilenet_v1,
                'mobilenet_v1_qc_se': mobilenet_v1_qc_se.mobilenet_v1,
                'mobilenet_v1_multilabel': mobilenet_v1_multilabel.mobilenet_v1,
                'nasnet_cifar': nasnet.build_nasnet_cifar,
                'nasnet_mobile': nasnet.build_nasnet_mobile,
                'nasnet_large': nasnet.build_nasnet_large,
                'nasnet_large_with_scope': nasnet.build_nasnet_large_with_scope,
                'pnasnet_large': pnasnet.build_pnasnet_large,
                'pnasnet_mobile': pnasnet.build_pnasnet_mobile,
                'pelee_net' : pelee_net.pelee_net,
                'pelee_net_multilabel' : pelee_net.pelee_net_multilabel,
                'shufflenet': shufflenet.shufflenet_g3,
                'shufflenet_multilabel': shufflenet_multilabel.shufflenet_g3,
                'shufflenet_v2': shufflenet_v2.shufflenet_v2,
                #'shufflenet_v2_multilabel': shufflenet_v2_multilabel.shufflenet_v2,
                'shufflenet_v2_multilabel': shufflenet_v2_multilabel.shufflenet_v2_d20,
                'shufflenet_v2d15_multilabel': shufflenet_v2_multilabel.shufflenet_v2_d15,
                'densenet_121_multilabel': densenet_multilabel.densenet_121,
                'densenet_169_multilabel': densenet_multilabel.densenet_169,
                'densenet_201_multilabel': densenet_multilabel.densenet_201,
                'densenet_264_multilabel': densenet_multilabel.densenet_264,
                # 'mobilenet_v3_large': mobilenet_v3.mobilenet_v3_large,
                # 'mobilenet_v3_small': mobilenet_v3.mobilenet_v3_small,
                # 'mobilenet_v3_large_new': mobilenet_v3_large.mobilenet,
                'efficientnet_b0': efficientnet_b0.mobilenet,
                'efficientnet_b0_wf': efficientnet_b0_wf.efficientnet_b0_wf,
                'mnasnet_a1': mnasnet_fix.mnasnet_a1,
                'mnasnet_b1': mnasnet.mnasnet_b1,
                'fairnas_a': fairnas_a.fairnas_a,
                'fairnas_b': fairnas_b.fairnas_b,
                'fairnas_c': fairnas_c.fairnas_c,
                'moga_a': moga_a.moga_a,
                'moga_b': moga_b.moga_b,
                'moga_c': moga_c.moga_c,
                'mobilenet_v2_qc': mobilenet_v2_qc.mobilenet,
                'mobilenet_v2_qc_multibranch': mobilenet_v2_qc_multibranch.mobilenet_two_branch,
                'mobilenet_v2_qc_multibranch_050': mobilenet_v2_qc_multibranch.mobilenet_two_branch_050,
                'mobilenet_v2_qc_multibranch_035': mobilenet_v2_qc_multibranch.mobilenet_two_branch_035,
                'mobilenet_v3_large': mobilenet_v3.large,
                'mobilenet_v3_small': mobilenet_v3.small,
                'VPN_Network': VPN_Network.base_net,
               }

arg_scopes_map = {'alexnet_v2': alexnet.alexnet_v2_arg_scope,
                  'cifarnet': cifarnet.cifarnet_arg_scope,
                  'overfeat': overfeat.overfeat_arg_scope,
                  'vgg_a': vgg.vgg_arg_scope,
                  'vgg_16': vgg.vgg_arg_scope,
                  'vgg_19': vgg.vgg_arg_scope,
                  'inception_v1': inception.inception_v3_arg_scope,
                  'inception_v2': inception.inception_v3_arg_scope,
                  'inception_v3': inception.inception_v3_arg_scope,
                  'inception_v4': inception.inception_v4_arg_scope,
                  'inception_resnet_v2': inception.inception_resnet_v2_arg_scope,
                  'inception_resnet_v2_center': inception.inception_resnet_v2_arg_scope,
                  'inception_resnet_v2_dropblock': inception.inception_resnet_v2_dropblock_arg_scope,
                  'lenet': lenet.lenet_arg_scope,
                  'resnet_v1_50': resnet_v1.resnet_arg_scope,
                  'resnet_v1_101': resnet_v1.resnet_arg_scope,
                  'resnet_v1_152': resnet_v1.resnet_arg_scope,
                  'resnet_v1_200': resnet_v1.resnet_arg_scope,
                  'resnet_v2_50': resnet_v2.resnet_arg_scope,
                  'resnet_v2_101': resnet_v2.resnet_arg_scope,
                  'resnet_v2_152': resnet_v2.resnet_arg_scope,
                  'resnet_v2_200': resnet_v2.resnet_arg_scope,
                  'mobilenet_v1': mobilenet_v1.mobilenet_v1_arg_scope,
                  'mobilenet_v1_075': mobilenet_v1.mobilenet_v1_arg_scope,
                  'mobilenet_v1_050': mobilenet_v1.mobilenet_v1_arg_scope,
                  'mobilenet_v1_025': mobilenet_v1.mobilenet_v1_arg_scope,
                  'mobilenet_v2': mobilenet_v2.training_scope,
                  'mobilenet_v2_multilabel': mobilenet_v2.training_scope,
                  'mobilenet_v2_035': mobilenet_v2.training_scope,
                  'mobilenet_v2_140': mobilenet_v2.training_scope,
                  'mobilenet_v2_140_multilabel': mobilenet_v2.training_scope,
                  'mobilenet_v1_qc': mobilenet_v1_qc.mobilenet_v1_arg_scope,
                  'mobilenet_v1_qc_mish': mobilenet_v1_qc_mish.mobilenet_v1_arg_scope,
                  'mobilenet_v1_qc_se': mobilenet_v1_qc_se.mobilenet_v1_arg_scope,
                  'mobilenet_v1_multilabel': mobilenet_v1_multilabel.mobilenet_v1_arg_scope,
                  'nasnet_cifar': nasnet.nasnet_cifar_arg_scope,
                  'nasnet_mobile': nasnet.nasnet_mobile_arg_scope,
                  'nasnet_large': nasnet.nasnet_large_arg_scope,
                  'nasnet_large_with_scope': nasnet.nasnet_large_arg_scope,
                  'pnasnet_large': pnasnet.pnasnet_large_arg_scope,
                  'pnasnet_mobile': pnasnet.pnasnet_mobile_arg_scope,
                  'pelee_net' : pelee_net.pelee_arg_scope,
                  'pelee_net_multilabel' : pelee_net.pelee_arg_scope,
                  'shufflenet': shufflenet.shufflenet_arg_scope,
                  'shufflenet_multilabel': shufflenet_multilabel.shufflenet_arg_scope,
                  'shufflenet_v2': shufflenet_v2.shufflenet_v2_arg_scope,
                  'shufflenet_v2_multilabel': shufflenet_v2_multilabel.shufflenet_v2_arg_scope,
                  'shufflenet_v2d15_multilabel': shufflenet_v2_multilabel.shufflenet_v2_arg_scope,
                  'densenet_121_multilabel': densenet_multilabel.densenet_arg_scope,
                  'densenet_169_multilabel': densenet_multilabel.densenet_arg_scope,
                  'densenet_201_multilabel': densenet_multilabel.densenet_arg_scope,
                  'densenet_264_multilabel': densenet_multilabel.densenet_arg_scope,
                  'efficientnet_b0': efficientnet_b0.training_scope,
                  'mobilenet_v3_large_new': mobilenet_v3_large.training_scope,
                  'mnasnet_b1': mnasnet.training_scope,
                  'mnasnet_a1': mnasnet_fix.training_scope,
                  'mobilenet_v2_qc': mobilenet_v2_qc.training_scope,
                  'mobilenet_v2_qc_multibranch': mobilenet_v2_qc_multibranch.training_scope,
                  'mobilenet_v2_qc_multibranch_050': mobilenet_v2_qc_multibranch.training_scope,
                  'mobilenet_v2_qc_multibranch_035': mobilenet_v2_qc_multibranch.training_scope,
                  'mobilenet_v3_large': mobilenet_v3.training_scope,
                  'mobilenet_v3_small': mobilenet_v3.training_scope,
                  'VPN_Network': VPN_Network.base_net_arg_scope,
                 }


def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False, num_classes_task2=None, **mkwargs):
    """Returns a network_fn such as `logits, end_points = network_fn(images)`.

    Args:
      name: The name of the network.
      num_classes: The number of classes to use for classification.
      weight_decay: The l2 coefficient for the model weights.
      is_training: `True` if the model is being used for training and `False`
        otherwise.

    Returns:
      network_fn: A function that applies the model to a batch of images. It has
        the following signature:
          logits, end_points = network_fn(images)
    Raises:
      ValueError: If network `name` is not recognized.
    """
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    func = networks_map[name]

    @functools.wraps(func)
    def network_fn(images, **kwargs):
        if name in arg_scopes_map:
            if num_classes_task2:
                arg_scope = arg_scopes_map[name](weight_decay=weight_decay, freeze_bn=True)
                kwargs['num_classes_task2'] = num_classes_task2
            else:
                arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
            kwargs.update(mkwargs)
            with slim.arg_scope(arg_scope):
                return func(images, num_classes, is_training=is_training, **kwargs)
        else:
            return func(images, num_classes, is_training=is_training, **kwargs)

    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn
