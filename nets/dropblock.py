
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import ops
import numbers

if tf.__version__.startswith("1.1."):
    BASE_LAYER = base._Layer
else:
    BASE_LAYER = base.Layer

def _bernoulli(shape, mean):
    return tf.nn.relu(tf.sign(mean - tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32)))

def nn_dropblock(x, keep_prob, block_size, noise_shape=None, seed=None, name=None):  # pylint: disable=invalid-name
  with ops.name_scope(name, "dropout", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
      raise ValueError("keep_prob must be a scalar tensor or a float in the "
                       "range (0, 1], got %g" % keep_prob)
    keep_prob = ops.convert_to_tensor(keep_prob,
                                      dtype=x.dtype,
                                      name="keep_prob")
    keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

    # Do nothing if we know keep_prob == 1
    if tensor_util.constant_value(keep_prob) == 1:
      return x

    batch_size, h, w, channel = x.get_shape().as_list()
    gamma = (1. - keep_prob) * (w * h) / (block_size ** 2) / \
            ((w - block_size + 1) * (h - block_size + 1))
    sampling_mask_shape = tf.stack([batch_size,
                                    h - block_size + 1,
                                    w - block_size + 1,
                                    channel])

    bottom = right = (block_size - 1) // 2
    top = left = (block_size - 1) - bottom
    padding = [[0, 0], [top, bottom], [left, right], [0, 0]]
    mask = _bernoulli(sampling_mask_shape, gamma)
    mask = tf.pad(mask, padding)
    mask = tf.nn.max_pool(mask, [1, block_size, block_size, 1], [1, 1, 1, 1], 'SAME')
    mask = 1 - mask
    output = x * mask
    return output


class DropBlock(BASE_LAYER):  # pylint: disable=protected-access

  def __init__(self, rate=0.5,
               block_size=3,
               noise_shape=None,
               seed=None,
               name=None,
               **kwargs):
      super(DropBlock, self).__init__(name=name, **kwargs)
      self.rate = rate
      self.noise_shape = noise_shape
      self.seed = seed
      self.block_size = block_size

  def call(self, inputs, training=False):
      def dropped_inputs():
          return nn_dropblock(inputs, self.rate, self.block_size,
                            noise_shape=self.noise_shape,
                            seed=self.seed)

      return utils.smart_cond(training,
                              dropped_inputs,
                              lambda: array_ops.identity(inputs))

def dropblock(inputs,
            rate=0.5,
            block_size=3,
            noise_shape=None,
            seed=None,
            is_training=False,
            name=None):
  layer = DropBlock(rate, block_size=block_size, noise_shape=noise_shape, seed=seed, name=name)
  return layer.apply(inputs, training=is_training)


def nn_dropout(x, keep_prob, noise_shape=None, seed=None, name=None):  # pylint: disable=invalid-name
  """Computes dropout.

  With probability `keep_prob`, outputs the input element scaled up by
  `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
  sum is unchanged.

  By default, each element is kept or dropped independently.  If `noise_shape`
  is specified, it must be
  [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
  will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
  and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
  kept independently and each row and column will be kept or not kept together.

  Args:
    x: A tensor.
    keep_prob: A scalar `Tensor` with the same type as x. The probability
      that each element is kept.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: A Python integer. Used to create random seeds. See
      @{tf.set_random_seed}
      for behavior.
    name: A name for this operation (optional).

  Returns:
    A Tensor of the same shape of `x`.

  Raises:
    ValueError: If `keep_prob` is not in `(0, 1]`.
  """
  with ops.name_scope(name, "dropout", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
      raise ValueError("keep_prob must be a scalar tensor or a float in the "
                       "range (0, 1], got %g" % keep_prob)
    keep_prob = ops.convert_to_tensor(keep_prob,
                                      dtype=x.dtype,
                                      name="keep_prob")
    keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

    # Do nothing if we know keep_prob == 1
    if tensor_util.constant_value(keep_prob) == 1:
      return x

    noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
    # uniform [keep_prob, 1.0 + keep_prob)
    random_tensor = keep_prob
    random_tensor += random_ops.random_uniform(noise_shape,
                                               seed=seed,
                                               dtype=x.dtype)
    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    binary_tensor = math_ops.floor(random_tensor)
    ret = math_ops.div(x, keep_prob) * binary_tensor
    ret.set_shape(x.get_shape())
    return ret

class Dropout(BASE_LAYER):  # pylint: disable=protected-access
  """Applies Dropout to the input.

  Dropout consists in randomly setting a fraction `rate` of input units to 0
  at each update during training time, which helps prevent overfitting.
  The units that are kept are scaled by `1 / (1 - rate)`, so that their
  sum is unchanged at training time and inference time.

  Arguments:
    rate: The dropout rate, between 0 and 1. E.g. `rate=0.1` would drop out
      10% of input units.
    noise_shape: 1D tensor of type `int32` representing the shape of the
      binary dropout mask that will be multiplied with the input.
      For instance, if your inputs have shape
      `(batch_size, timesteps, features)`, and you want the dropout mask
      to be the same for all timesteps, you can use
      `noise_shape=[batch_size, 1, features]`.
    seed: A Python integer. Used to create random seeds. See
      @{tf.set_random_seed}.
      for behavior.
    name: The name of the layer (string).
  """

  def __init__(self, rate=0.5,
               noise_shape=None,
               seed=None,
               name=None,
               **kwargs):
    super(Dropout, self).__init__(name=name, **kwargs)
    self.rate = rate
    self.noise_shape = noise_shape
    self.seed = seed

  def call(self, inputs, training=False):
    def dropped_inputs():
      return nn_dropout(inputs, 1  - self.rate,
                        noise_shape=self.noise_shape,
                        seed=self.seed)
    return utils.smart_cond(training,
                            dropped_inputs,
                            lambda: array_ops.identity(inputs))


def dropout(inputs,
            rate=0.5,
            noise_shape=None,
            seed=None,
            is_training=False,
            name=None):
  """Applies Dropout to the input.

  Dropout consists in randomly setting a fraction `rate` of input units to 0
  at each update during training time, which helps prevent overfitting.
  The units that are kept are scaled by `1 / (1 - rate)`, so that their
  sum is unchanged at training time and inference time.

  Arguments:
    inputs: Tensor input.
    rate: The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out
      10% of input units.
    noise_shape: 1D tensor of type `int32` representing the shape of the
      binary dropout mask that will be multiplied with the input.
      For instance, if your inputs have shape
      `(batch_size, timesteps, features)`, and you want the dropout mask
      to be the same for all timesteps, you can use
      `noise_shape=[batch_size, 1, features]`.
    seed: A Python integer. Used to create random seeds. See
      @{tf.set_random_seed}
      for behavior.
    training: Either a Python boolean, or a TensorFlow boolean scalar tensor
      (e.g. a placeholder). Whether to return the output in training mode
      (apply dropout) or in inference mode (return the input untouched).
    name: The name of the layer (string).

  Returns:
    Output tensor.
  """
  layer = Dropout(rate, noise_shape=noise_shape, seed=seed, name=name)
  return layer.apply(inputs, training=is_training)