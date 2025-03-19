#!/data/anaconda2/bin/python
# coding=utf-8
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
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.python.ops import control_flow_ops
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
from tensorflow.python.framework import ops
from autoaugment import ImageNetPolicy

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'max_ckpt_to_keep', 1000,
    'max_ckpt_to_keep.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

tf.app.flags.DEFINE_integer(
    'quantize_delay', -1,
    'Number of steps to start quantized training. Set to -1 would disable '
    'quantized training.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_string(
    'learning_rate_conf_file',
    None,
    'Specifies learning rate from conf file')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer(
    'train_image_count', None, 'Train image count')

tf.app.flags.DEFINE_integer(
    'label_count', None, 'Label count')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

tf.app.flags.DEFINE_string(
    'tf_file_pattern', '', 'tf_file_pattern.')

# add@20190605
tf.app.flags.DEFINE_boolean('rotate_image', False,
                            'Random rotate the image during training.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

#####################
# Center_loss Flags #
#####################
tf.app.flags.DEFINE_float(
    'center_loss_factor', 0.1,
    'Center loss factor.')

tf.app.flags.DEFINE_float(
    'center_loss_alfa', 0.9,
    'Center update rate for center loss.')

tf.app.flags.DEFINE_boolean(
    'add_center_loss', False,
    'whether add center loss.')

# TensorFlow Server模型描述信息，包括作业名称，任务编号，隐含层神经元数量，MNIST数据目录以及每次训练数据大小（默认一个批次为100个图片）
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

### add by ffwang, 某些类别的随机裁剪比例不一样
tf.app.flags.DEFINE_string('special_labels_4_crop_range', None,
                           'labels that need special crop range in preprocessing.')

tf.app.flags.DEFINE_float("crop_area_min", 0.05, "crop_area_min for preprocessing")

# mix up wangfei
tf.app.flags.DEFINE_bool('mixup', True, 'mixup training')
tf.app.flags.DEFINE_float("mixup_alpha", 0.5, "mixup_alpha_random")
tf.app.flags.DEFINE_float("mixup_percent", 0.5, "mixup_percent")

# Define autoaug
tf.app.flags.DEFINE_bool('autoaug', False, 'autoaug or not')

FLAGS = tf.app.flags.FLAGS


def _configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      ValueError: if
    """
    if FLAGS.learning_rate_conf_file:
        lines = open(FLAGS.learning_rate_conf_file).read().splitlines()
        tf.logging.info("************************** FLAGS.learning_rate_conf_file:     %s",
                        FLAGS.learning_rate_conf_file)
        tf.logging.info("************************** len(lines):     %s", len(lines))
        if len(lines) == 0:
            raise ValueError('learning_rate_conf_file INVALID!')

        learning_rate_conf_array_lr = []
        learning_rate_conf_array_step = []
        for line in lines:
            sep = line.split(":")
            if len(sep) == 2:
                learning_rate_conf_array_lr.append(float(sep[1]))
                learning_rate_conf_array_step.append(int(sep[0]))
        tf.logging.info("************************** learning_rate_conf_array_step: %s", learning_rate_conf_array_step)
        tf.logging.info("************************** learning_rate_conf_array_lr:   %s", learning_rate_conf_array_lr)
        tensor_learning_rate_conf_array_step = tf.convert_to_tensor(learning_rate_conf_array_step, dtype=tf.int64)
        tensor_learning_rate_conf_array_lr = tf.convert_to_tensor(learning_rate_conf_array_lr, dtype=tf.float32)
        tf.logging.info("************************** tensor_learning_rate_conf_array_step: %s",
                        tensor_learning_rate_conf_array_step)
        tf.logging.info("************************** tensor_learning_rate_conf_array_lr:   %s",
                        tensor_learning_rate_conf_array_lr)

        def get_learning_rate(global_step):
            global_step_tile = tf.tile([global_step], [tensor_learning_rate_conf_array_step.shape.as_list()[0]])
            print("global_step_tile:", global_step_tile)
            idx = tf.where(global_step_tile <= tensor_learning_rate_conf_array_step)[0][0]
            return tf.gather(learning_rate_conf_array_lr, [idx])[0]

        return get_learning_rate(global_step)

    decay_steps = int(num_samples_per_epoch * FLAGS.num_epochs_per_decay /
                      FLAGS.batch_size / FLAGS.num_clones)

    if FLAGS.max_number_of_steps and  decay_steps > FLAGS.max_number_of_steps:
        decay_steps = FLAGS.max_number_of_steps

    if FLAGS.sync_replicas:
        decay_steps /= FLAGS.replicas_to_aggregate


    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.

    Args:
      learning_rate: A scalar or `Tensor` learning rate.

    Returns:
      An instance of an optimizer.

    Raises:
      ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=FLAGS.adadelta_rho,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=FLAGS.ftrl_learning_rate_power,
            initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
            l1_regularization_strength=FLAGS.ftrl_l1,
            l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.rmsprop_momentum,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer


def _add_variables_summaries(learning_rate):
    summaries = []
    for variable in slim.get_model_variables():
        summaries.append(tf.summary.histogram(variable.op.name, variable))
    summaries.append(tf.summary.scalar('training/Learning Rate', learning_rate))
    return summaries


def _get_init_fn():
    """Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """
    if FLAGS.checkpoint_path is None:
        return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(FLAGS.train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % FLAGS.train_dir)
        return None

    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                break
        else:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)
    tf.logging.info('len(variables_to_restore): %d' % len(variables_to_restore))

    if len(variables_to_restore):
        return slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore,
            ignore_missing_vars=FLAGS.ignore_missing_vars)
    else:
        variables_in_model_all = slim.get_variables_to_restore()
        tf.logging.info('len(variables_in_model_all): %d' % len(variables_in_model_all))

        variables_to_restore = []
        for var in variables_in_model_all:
            exclu = False
            for exclusion in exclusions:
                # print(exclusion)
                # print(var.op.name)
                if var.op.name.startswith(exclusion):
                    print('===== exclused variable: ', var.op.name)
                    exclu = True
                    break
            if not exclu:
                variables_to_restore.append(var)

        return slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore,
            ignore_missing_vars=FLAGS.ignore_missing_vars)


# add center_loss
def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers


def _get_variables_to_train():
    """Returns a list of variables to train.

    Returns:
      A list of variables to train by the optimizer.
    """
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train

def getMixupData(images, labels):
    dice_random = tf.random_uniform([], 0, 1, dtype=tf.float32, seed=None)

    def do_mixup():
        weight = np.random.beta(FLAGS.mixup_alpha, FLAGS.mixup_alpha, FLAGS.batch_size)
        x_weight = weight.reshape(FLAGS.batch_size, 1, 1, 1)
        y_weight = weight.reshape(FLAGS.batch_size, 1)
        index = np.random.permutation(FLAGS.batch_size)
        x1, x2 = images, tf.gather(images, index)
        x = x1 * x_weight + x2 * (1 - x_weight)
        y1, y2 = labels, tf.gather(labels, index)
        y = y1 * y_weight + y2 * (1 - y_weight)
        return x, y

    def do_nothing():
        return images, labels

    return tf.cond(tf.less_equal(dice_random, FLAGS.mixup_percent), do_mixup, do_nothing)


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        #######################
        # Config model_deploy #
        #######################

        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=FLAGS.task,
            num_replicas=FLAGS.worker_replicas,
            num_ps_tasks=FLAGS.num_ps_tasks)

        # Create global_step
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        ######################
        # Select the dataset #
        ######################
        tf_file_pattern = None
        if FLAGS.tf_file_pattern:
            tf_file_pattern = FLAGS.tf_file_pattern
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir,
            train_image_count=FLAGS.train_image_count, label_count=FLAGS.label_count, file_pattern=tf_file_pattern)
        tf.logging.info("************************** tf_file_pattern:     %s", tf_file_pattern)
        tf.logging.info("************************** dataset.num_classes: %s", dataset.num_classes)
        tf.logging.info("************************** dataset.num_samples: %s", dataset.num_samples)
        tf.logging.info("************************** FLAGS.learning_rate: %s", FLAGS.learning_rate)
        tf.logging.info("************************** FLAGS.learning_rate_decay_factor: %s",
                        FLAGS.learning_rate_decay_factor)
        tf.logging.info("************************** FLAGS.num_epochs_per_decay: %s", FLAGS.num_epochs_per_decay)
        tf.logging.info("************************** FLAGS.special_labels_4_crop_range: %s",
                        FLAGS.special_labels_4_crop_range)
        tf.logging.info("************************** FLAGS.add_center_loss: %s", FLAGS.add_center_loss)
        tf.logging.info("************************** FLAGS.rotate_image: %s", FLAGS.rotate_image)
        special_labels_list = []
        if FLAGS.special_labels_4_crop_range is not None:
            special_labels_list = FLAGS.special_labels_4_crop_range.split(",")
            special_labels_list = [int(item) for item in special_labels_list]
        tf.logging.info("************************** special_labels_list: %s", special_labels_list)
        tf.logging.info("************************** crop_area_min: %f", FLAGS.crop_area_min)
        tf.logging.info("************************** quantize_delay: %d", FLAGS.quantize_delay)
        tf.logging.info("************************** mixup:         %s", FLAGS.mixup)
        tf.logging.info("************************** mixup_alpha:   %s", FLAGS.mixup_alpha)
        tf.logging.info("************************** mixup_percent: %s", FLAGS.mixup_percent)
        ######################
        # Select the network #
        ######################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            weight_decay=FLAGS.weight_decay,
            is_training=True)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=True)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        with tf.device(deploy_config.inputs_device()):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=FLAGS.num_readers,
                common_queue_capacity=20 * FLAGS.batch_size,
                common_queue_min=10 * FLAGS.batch_size)
            [image, label] = provider.get(['image', 'label'])
            label -= FLAGS.labels_offset

            train_image_size = FLAGS.train_image_size or network_fn.default_image_size
            tf.logging.info("************************** train_image_size: %s", train_image_size)

            if FLAGS.autoaug:
                tf.logging.info("************************** using autoaug:")
                print("======using autoaug")
                image = tf.py_func(ImageNetPolicy(), [image], tf.uint8)

            image = image_preprocessing_fn(image, train_image_size, train_image_size, label=label,
                                           special_labels_list=special_labels_list, crop_area_min=FLAGS.crop_area_min,
                                           rotate_image=FLAGS.rotate_image)

            '''
            tensor_special_labels_list = tf.convert_to_tensor(special_labels_list, dtype=tf.int64)
            tf.logging.info("************************** tensor_special_labels_list: %s", tensor_special_labels_list)
      
            # 对于指定的labelid，crop时使用特殊的参数
            def normal_image_preprocessing_fn():
                return image_preprocessing_fn(image, train_image_size, train_image_size)
            def special_image_preprocessing_fn():
                return image_preprocessing_fn(image, train_image_size, train_image_size, crop_area_range=(0.81, 1.0))
            image = tf.cond(tf.equal(tf.reduce_sum(tf.cast(tf.equal(tensor_special_labels_list, tf.ones(tensor_special_labels_list.shape, dtype=tf.int64)*label), tf.int64)), 1),
                            special_image_preprocessing_fn, normal_image_preprocessing_fn)
            '''

            images, labels = tf.train.batch(
                [image, label],
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size)
            print("labels:", labels)
            centerlabel = labels
            labels = slim.one_hot_encoding(
                labels, dataset.num_classes - FLAGS.labels_offset)
            print("labels:", labels)

            # add mix up
            if FLAGS.mixup:
                tf.logging.info("************************** labels before mixup: %s", labels)
                print("====using mixup")
                images, labels = getMixupData(images, labels)
            tf.logging.info("************************** labels: %s", labels)

            batch_queue = slim.prefetch_queue.prefetch_queue(
                [images, labels, centerlabel], capacity=2 * deploy_config.num_clones)

        ####################
        # Define the model #
        ####################
        def clone_fn(batch_queue):
            """Allows data parallelism by creating multiple clones of network_fn."""
            images, labels, centerlabel = batch_queue.dequeue()
            logits, end_points = network_fn(images)

            #############################
            # Specify the loss function #
            #############################
            if 'AuxLogits' in end_points:
                tf.losses.softmax_cross_entropy(
                    logits=end_points['AuxLogits'], onehot_labels=labels,
                    label_smoothing=FLAGS.label_smoothing, weights=0.4, scope='aux_loss')
            tf.losses.softmax_cross_entropy(
                logits=logits, onehot_labels=labels,
                label_smoothing=FLAGS.label_smoothing, weights=1.0)

            if FLAGS.add_center_loss and 'CenterLogits' in end_points:
                tf.logging.info("************************** Center_Loss **************************")
                centerlogit = end_points['CenterLogits']

                centerloss, centers = center_loss(centerlogit, centerlabel, FLAGS.center_loss_alfa,
                                                  dataset.num_classes - FLAGS.labels_offset)
                centerloss_final = tf.multiply(centerloss, FLAGS.center_loss_factor, 'center_loss')
                ops.add_to_collection(tf.GraphKeys.LOSSES, centerloss_final)

            return end_points

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by network_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        # Add summaries for end_points.
        end_points = clones[0].outputs
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x)))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        #################################
        # Configure the moving averages #
        #################################
        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        if FLAGS.quantize_delay >= 0:
            tf.contrib.quantize.create_training_graph(
                quant_delay=FLAGS.quantize_delay)

        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
            optimizer = _configure_optimizer(learning_rate)
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        if FLAGS.sync_replicas:
            # If sync_replicas is enabled, the averaging will be done in the chief
            # queue runner.
            optimizer = tf.train.SyncReplicasOptimizer(
                opt=optimizer,
                replicas_to_aggregate=FLAGS.replicas_to_aggregate,
                variable_averages=variable_averages,
                variables_to_average=moving_average_variables,
                # replica_id=tf.constant(FLAGS.task, tf.int32, shape=()),
                total_num_replicas=FLAGS.worker_replicas)
        elif FLAGS.moving_average_decay:
            # Update ops executed locally by trainer.
            update_ops.append(variable_averages.apply(moving_average_variables))

        # Variables to train.
        variables_to_train = _get_variables_to_train()

        #  and returns a train_tensor and summary_op
        total_loss, clones_gradients = model_deploy.optimize_clones(
            clones,
            optimizer,
            var_list=variables_to_train)
        # Add total_loss to summary.
        summaries.add(tf.summary.scalar('total_loss', total_loss))

        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(clones_gradients,
                                                 global_step=global_step)
        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)
        train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                                          name='train_op')

        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or _gather_clone_loss().

        ## 添加了条件crop后，summary报错，注释这里，只保留主要的summary. ffwang 20180122
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                           first_clone_scope))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        # Allow the GPU memory growth.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Set the maximum number of the recent checkpoint files to 100.
        saver = tf.train.Saver(max_to_keep=FLAGS.max_ckpt_to_keep)

        ###########################
        # Kicks off the training. #
        ###########################
        slim.learning.train(
            train_tensor,
            logdir=FLAGS.train_dir,
            master=FLAGS.master,
            is_chief=(FLAGS.task == 0),
            init_fn=_get_init_fn(),
            summary_op=summary_op,
            number_of_steps=FLAGS.max_number_of_steps,
            log_every_n_steps=FLAGS.log_every_n_steps,
            save_summaries_secs=FLAGS.save_summaries_secs,
            save_interval_secs=FLAGS.save_interval_secs,
            sync_optimizer=optimizer if FLAGS.sync_replicas else None,
            saver=saver,
            session_config=config)


if __name__ == '__main__':
    tf.app.run()
