import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.moga.ops import conv2d_bn,separable_conv,res_block,conv2d_bias, conv2d


def moga_c(inputs, num_classes, is_training=False):
    config=[[3, 24, 5, 2, 'RE', False],
            [3, 24, 3, 1, 'RE', False],
            [3, 40, 5, 2, 'RE', False],
            [3, 40, 3, 1, 'RE', False],
            [3, 40, 5, 1, 'RE', False],
            [3, 80, 5, 2, 'HS', False],
            [6, 80, 5, 1, 'HS', True],
            [3, 80, 5, 1, 'HS', False],
            [3, 80, 5, 1, 'HS', False],
            [6, 112, 3, 1, 'HS', False],
            [6, 112, 3, 1, 'HS', True],
            [6, 160, 3, 2, 'HS', True],
            [6, 160, 3, 1, 'HS', True],
            [6, 160, 3, 1, 'HS', True]]

    with tf.variable_scope('moga'):
        end_points = {}
        net = conv2d_bn(inputs, 16, 3, 2, is_training, scope='stem', non_linear='HS')
        net = separable_conv(net, 16, 3, 1, is_training, scope='mb_1')

        for i, (e, oc, k, s, nl, se) in enumerate(config):
            net = res_block(net, e, oc, k, s, is_training, scope='mb_%s'%(i+2), non_linear=nl, squeeze_excitation=se)

        net = conv2d_bn(net, 960, 1, 1, is_training, scope='conv_before_pooling', non_linear='HS')
        net = slim.avg_pool2d(net, kernel_size=7)
        net = conv2d(net, 1280, 1, 1, scope="conv_after_pooling", non_linear='HS')
        net = conv2d_bias(net, num_classes, 1, 1, scope="conv_after_pooling2", non_linear=None)
        logits = tf.squeeze(net, axis=[1, 2])
        end_points['Logits'] = logits
        end_points['Predictions_sigmoid'] = tf.nn.sigmoid(logits, name='Predictions_sigmoid')
        end_points['Predictions_softmax'] = tf.nn.softmax(logits, name='Predictions_softmax')
        return logits, end_points


if __name__ == "__main__":
    inputs = tf.ones(shape=[1, 224, 224, 3], dtype=tf.float32)
    net = moga_c(inputs,34,False)