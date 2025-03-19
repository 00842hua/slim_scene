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

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy. Note For '
                            'historical reasons loss from all clones averaged '
                            'out and learning rate decay happen per clone '
                            'epochs')

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
    '"polynomial", or "cosine"')

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
    'Number of epochs after which learning rate decays. Note: this flag counts '
    'epochs per clone but aggregates per sync replicas. So 1.0 means that '
    'each clone will go over full epoch individually, but replicas will go '
    'once across all replicas.')

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

# tf.app.flags.DEFINE_integer(
#     'train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', 180000,
                            'The maximum number of training steps.')

tf.app.flags.DEFINE_integer(
    'train_image_count', None, 'Train image count')

tf.app.flags.DEFINE_integer(
    'label_count', None, 'Label count')

tf.app.flags.DEFINE_string(
    'tf_file_pattern', '', 'tf_file_pattern.')

tf.app.flags.DEFINE_float("crop_area_min", 0.05, "crop_area_min for preprocessing")


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

#mix up wangfei
tf.app.flags.DEFINE_bool('mixup', False, 'mixup training')
tf.app.flags.DEFINE_float("mixup_alpha", 0.5, "mixup_alpha_random")
tf.app.flags.DEFINE_float("mixup_percent", 0.5, "mixup_percent")

#distill wangfei
# tf.app.flags.DEFINE_string(
#     'distill_model_name', None, 'The teacher model.')

tf.app.flags.DEFINE_string(
    'distill_model_ckpt', None, 'The teacher model ckpt')

tf.app.flags.DEFINE_string(
    'loss_type', 'mse', 'loss type should in [mse, mpse, sigmoid]')


tf.app.flags.DEFINE_integer(
    'train_image_size_teacher_L', None, 'train_image_size_teacher_L')

tf.app.flags.DEFINE_integer(
    'train_image_size_student_S', None, 'train_image_size_student_S')


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
  # Note: when num_clones is > 1, this will actually have each clone to go
  # over each epoch FLAGS.num_epochs_per_decay times. This is different
  # behavior from sync replicas and is expected to produce different results.

  if FLAGS.learning_rate_conf_file:
      lines = open(FLAGS.learning_rate_conf_file).read().splitlines()
      tf.logging.info("************************** FLAGS.learning_rate_conf_file:     %s", FLAGS.learning_rate_conf_file)
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
  if decay_steps > FLAGS.max_number_of_steps:
    decay_steps = FLAGS.max_number_of_steps

  if FLAGS.sync_replicas:
    decay_steps /= FLAGS.replicas_to_aggregate

  tf.logging.info("************************** FLAGS.learning_rate:              %s", FLAGS.learning_rate)
  tf.logging.info("************************** FLAGS.learning_rate_decay_factor: %s", FLAGS.learning_rate_decay_factor)
  tf.logging.info("************************** decay_steps:                      %s", decay_steps)
  tf.logging.info("************************** global_step:                      %s", global_step)
  tf.logging.info("************************** num_samples_per_epoch:            %s", num_samples_per_epoch)

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
  elif FLAGS.learning_rate_decay_type == 'cosine':
    return tf.train.cosine_decay(FLAGS.learning_rate,
                                 global_step,
                                 decay_steps,
                                 alpha=0.0,
                                 name='cosine_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized' %
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
        name='Momentum',
        use_nesterov=True)
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized' % FLAGS.optimizer)
  return optimizer


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
  distill_variables_to_restore = []
  for var in slim.get_model_variables():
    if var.op.name.startswith("ssd_300_vgg/"):
        if -1 == var.op.name.find("_quant"):
            distill_variables_to_restore.append(var)
    else:
      excluded = False
      for exclusion in exclusions:
        if var.op.name.startswith(exclusion) or -1 != var.op.name.find("_quant"):
            tf.logging.info("exclude: %s|%s" % (exclusion, var.op.name))
            excluded = True
            break
      if not excluded:
        variables_to_restore.append(var)

  if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
  else:
    checkpoint_path = FLAGS.checkpoint_path

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)

  tf.logging.info("************************** len(distill_variables_to_restore): %d", len(distill_variables_to_restore))
  for var in distill_variables_to_restore:
      print(var.op.name)
  tf.logging.info("************************** len(variables_to_restore): %d", len(variables_to_restore))
  for var in variables_to_restore:
      print(var.op.name)

  distill_saver = tf.train.Saver(distill_variables_to_restore)
  saver = tf.train.Saver(variables_to_restore)

  def callback(session):
      distill_saver.restore(session, FLAGS.distill_model_ckpt)
      saver.restore(session, checkpoint_path)

  return callback



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


def getMixupData(imagesL, imagesS, labels):
    dice_random = tf.random_uniform([], 0, 1, dtype=tf.float32, seed=None)
    def do_mixup():
        weight = np.random.beta(FLAGS.mixup_alpha, FLAGS.mixup_alpha, FLAGS.batch_size)
        x_weight = weight.reshape(FLAGS.batch_size, 1, 1, 1)
        y_weight = weight.reshape(FLAGS.batch_size, 1)
        index = np.random.permutation(FLAGS.batch_size)
        xL1, xL2 = imagesL, tf.gather(imagesL, index)
        xL = xL1 * x_weight + xL2 * (1 - x_weight)
        xS1, xS2 = imagesS, tf.gather(imagesS, index)
        xS = xS1 * x_weight + xS2 * (1 - x_weight)
        y1, y2 = labels, tf.gather(labels, index)
        y = y1 * y_weight + y2 * (1 - y_weight)
        return xL, xS, y
    def do_nothing():
        return imagesL, imagesS, labels
    return tf.cond(tf.less_equal(dice_random, FLAGS.mixup_percent), do_mixup, do_nothing)


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  if not FLAGS.train_image_size_teacher_L:
    raise ValueError('You must supply --train_image_size_teacher_L')
  if not FLAGS.train_image_size_student_S:
    raise ValueError('You must supply --train_image_size_student_S')
  # if not FLAGS.distill_model_name:
  #   raise ValueError('You must supply --distill_model_name')
  if not FLAGS.distill_model_ckpt:
    raise ValueError('You must supply --distill_model_ckpt')


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
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir,
        train_image_count=FLAGS.train_image_count, label_count=FLAGS.label_count, file_pattern=FLAGS.tf_file_pattern)
    tf.logging.info("************************** tf_file_pattern:     %s", FLAGS.tf_file_pattern)
    tf.logging.info("************************** label_count:         %s", FLAGS.label_count)
    tf.logging.info("************************** train_image_count:   %s", FLAGS.train_image_count)
    tf.logging.info("************************** mixup:         %s", FLAGS.mixup)
    tf.logging.info("************************** mixup_alpha:   %s", FLAGS.mixup_alpha)
    tf.logging.info("************************** mixup_percent: %s", FLAGS.mixup_percent)
    # tf.logging.info("************************** distill_model_name: %s", FLAGS.distill_model_name)
    tf.logging.info("************************** distill_model_ckpt: %s", FLAGS.distill_model_ckpt)
    tf.logging.info("************************** train_image_size_teacher_L: %s", FLAGS.train_image_size_teacher_L)
    tf.logging.info("************************** train_image_size_student_S: %s", FLAGS.train_image_size_student_S)

    ######################
    # Select the network #
    ######################

    distill_network_fn = nets_factory.get_network_fn(
        "VPN_Network",
        num_classes=895,
        is_training=False)
            
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
      #label -= FLAGS.labels_offset

      # train_image_size = FLAGS.train_image_size or network_fn.default_image_size

      train_image_size_teacher_L = FLAGS.train_image_size_teacher_L
      train_image_size_student_S = FLAGS.train_image_size_student_S
      tf.logging.info("************************** image:   %s", image)
      imageL, imageS = image_preprocessing_fn(image, train_image_size_teacher_L, train_image_size_student_S)
      tf.logging.info("************************** imageS:   %s", imageS)
      tf.logging.info("************************** imageL:   %s", imageL)

      # tf.logging.info("************************** imageL: %s", imageL)
      # tf.logging.info("************************** imageS: %s", imageS)

      imagesL, imagesS, labels = tf.train.batch(
          [imageL, imageS, label],
          batch_size=FLAGS.batch_size,
          num_threads=FLAGS.num_preprocessing_threads,
          capacity=5 * FLAGS.batch_size)
      #labels = slim.one_hot_encoding(
      #    labels, dataset.num_classes - FLAGS.labels_offset)
      #########################################################################
      # mixup implementation
      #weight = np.random.beta(0.2, 0.2, FLAGS.batch_size)
      #index = np.random.permutation(FLAGS.batch_size)

      #x1, y1 = images, labels
      #x2 = tf.gather(images, index)
      #y2 = tf.gather(labels, index)
      #x1 = tf.transpose(x1, perm=[1, 2, 3, 0])
      #x2 = tf.transpose(x2, perm=[1, 2, 3, 0])
      #y1 = tf.transpose(y1)
      #y2 = tf.transpose(y2)
      #w = tf.convert_to_tensor(weight, dtype=tf.float32)
      #x = tf.multiply(w, x1) + tf.multiply(1 - w, x2)
      #y = tf.multiply(w, y1) + tf.multiply(1 - w, y2)
      #x = tf.transpose(x, perm=[3, 0, 1, 2])
      #y = tf.transpose(y)
      #########################################################################

      #add mix up
      if FLAGS.mixup:
        tf.logging.info("************************** labels before mixup: %s", labels)
        imagesL, imagesS, labels = getMixupData(imagesL, imagesS, labels)
      tf.logging.info("************************** labels: %s", labels)
      
      batch_queue = slim.prefetch_queue.prefetch_queue(
          [imagesL, imagesS, labels], capacity=2 * deploy_config.num_clones)
      #batch_queue = slim.prefetch_queue.prefetch_queue(
      #    [x, y], capacity=2 * deploy_config.num_clones)

    ####################
    # Define the model #
    ####################
    def clone_fn(batch_queue):
      """Allows data parallelism by creating multiple clones of network_fn."""
      imagesL, imagesS, labels = batch_queue.dequeue()
      tf.logging.info("************************** imagesL:   %s", imagesL)
      logits, end_points = network_fn(imagesS)

      distill_logits, _, _, _ = distill_network_fn(imagesL)
      tf.logging.info("************************** distill_logits: %s", distill_logits)
      tf.logging.info("************************** logits:         %s", logits)
      tf.logging.info("************************** FLAGS.loss_type: %s", FLAGS.loss_type)

      if FLAGS.loss_type == 'mse':
          tf.logging.info("************************** mse loss **************************")
          mse_loss = tf.reduce_mean(tf.square(tf.subtract(logits, distill_logits)))
          # mse_loss = tf.losses.mean_squared_error(logits, distill_logits)
          tf.losses.add_loss(mse_loss)
      elif FLAGS.loss_type == 'mpse':
          tf.logging.info("************************** mpse loss **************************")

          mpse_loss = tf.Variable(0.0, dtype=tf.float32)
          i = tf.constant(1)
          n = tf.constant(logits.get_shape().as_list()[1])
          def cond(i,n,mpse_loss):
              return i < n
          def body(i, n, mpse_loss):
              logits_shift = tf.roll(logits, shift=i, axis=1)
              distill_logits_shift = tf.roll(distill_logits, shift=i, axis=1)
              logits_diff = logits - logits_shift
              distill_logits_diff = distill_logits - distill_logits_shift
              curr_loss = tf.reduce_mean(tf.square(tf.subtract(logits_diff, distill_logits_diff)))
              mpse_loss += curr_loss
              i = i + 1

              # tf.logging.info("************************** logits_shift")
              # tf.logging.info(logits_shift)
              # tf.logging.info("************************** logits_diff")
              # tf.logging.info(logits_diff)
              # tf.logging.info("************************** distill_logits_shift")
              # tf.logging.info(distill_logits_shift)
              # tf.logging.info("************************** curr_loss")
              # tf.logging.info(curr_loss)
              # tf.logging.info("************************** mpse_loss")
              # tf.logging.info(mpse_loss)
              # tf.logging.info("************************** i")
              # tf.logging.info(i)
              return i, n, mpse_loss

          i, n, mpse_loss = tf.while_loop(cond, body, [i, n, mpse_loss])

          mpse_loss = mpse_loss * 2 / (tf.cast(n, dtype=tf.float32) - tf.constant(1.0, dtype=tf.float32))
          tf.losses.add_loss(mpse_loss)
      else:
          tf.logging.info("************************** sigmoid_cross_entropy loss **************************")
          distill_prediction = tf.nn.sigmoid(distill_logits)
          tf.losses.sigmoid_cross_entropy(
              distill_prediction, logits, label_smoothing=FLAGS.label_smoothing, weights=1.0)

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
      learning_rate = _configure_learning_rate(FLAGS.train_image_count, global_step)
      optimizer = _configure_optimizer(learning_rate)
      summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    if FLAGS.sync_replicas:
      # If sync_replicas is enabled, the averaging will be done in the chief
      # queue runner.
      optimizer = tf.train.SyncReplicasOptimizer(
          opt=optimizer,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          total_num_replicas=FLAGS.worker_replicas,
          variable_averages=variable_averages,
          variables_to_average=moving_average_variables)
    elif FLAGS.moving_average_decay:
      # Update ops executed locally by trainer.
      update_ops.append(variable_averages.apply(moving_average_variables))

    # Variables to train.
    variables_to_train = _get_variables_to_train()

    # do NOT train teacher model!!
    variables_to_train_real = []
    for var in variables_to_train:
        if var.op.name.startswith("ssd_300_vgg/"):
            pass
        else:
            variables_to_train_real.append(var)
    tf.logging.info("************************** len(variables_to_train): %d", len(variables_to_train))
    tf.logging.info("************************** len(variables_to_train_real): %d", len(variables_to_train_real))
    # for var in variables_to_train_real:
    #   print(var.op.name)
    variables_to_train = variables_to_train_real

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
    with tf.control_dependencies([update_op]):
      train_tensor = tf.identity(total_loss, name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    # Allow the GPU memory growth.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Set the maximum number of the recent checkpoint files to 100.
    saver = tf.train.Saver(max_to_keep=100)

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
