#-*- coding:utf-8 -*-

import tensorflow as tf
import sys
import keras.backend as K
import matplotlib.pyplot as plt
import cv2
import io
import os

sys.path.append("../")

from preprocessing import preprocessing_factory
import numpy as np
import scipy.misc

PY3FLAG = sys.version_info > (3,0)
if PY3FLAG:
    open_fn = open
else:
    import codecs
    open_fn = codecs.open

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('imagelist_path', None, 'image to predict')
tf.app.flags.DEFINE_string('pb_path', None, 'pb file')
tf.app.flags.DEFINE_string('labels_file', None, 'labels file')
tf.app.flags.DEFINE_string('output_node_names', None, 'output_node_names')
tf.app.flags.DEFINE_integer('image_size', 224, 'image_size.')
tf.app.flags.DEFINE_string('preprocess_type', 'inception_common', 'preprocess type')
tf.app.flags.DEFINE_string('last_node_names',None,'last_node_names')
tf.app.flags.DEFINE_integer('last_node_size', 1024, 'last_node_size.')
tf.app.flags.DEFINE_string('out_dir',None,'out_dir')


if __name__ == "__main__":
    if not FLAGS.pb_path:
        raise ValueError('need set pb_path')

    if not FLAGS.imagelist_path:
        raise ValueError('need set imagelist_path')

    if not FLAGS.labels_file:
        raise ValueError('need set labels_file')

    if not FLAGS.output_node_names:
        raise ValueError('need set output_node_names')

    if not FLAGS.preprocess_type:
        raise ValueError('need set preprocess_type')

    if not FLAGS.last_node_names:
        raise ValueError('need set last_node_names')

    graph_def = tf.GraphDef()

    with open(FLAGS.pb_path, 'rb') as f:
        graph_def.ParseFromString(f.read())

    tf.import_graph_def(graph_def, name='')

    sess = tf.Session()

    image_preprocessing_fn = preprocessing_factory.get_preprocessing(FLAGS.preprocess_type, is_training=False)
    #image_data = tf.gfile.FastGFile(FLAGS.image_path, 'rb').read()

    image_list = []
    output_dir = "/home/ypf/data/heatmap_data/heatmap"
    if FLAGS.out_dir:
        output_dir = FLAGS.out_dir
    
    for image_path in io.open(FLAGS.imagelist_path,encoding='utf-8').read().splitlines():

        image_name = image_path.split("/")[-1]

        print(image_name)

        with open(image_path, 'rb') as f:
            image_data = f.read()

        image_data_tensor = tf.image.decode_jpeg(image_data)

        preprocessed = image_preprocessing_fn(image_data_tensor, FLAGS.image_size, FLAGS.image_size)

        preprocessed_reshape = tf.expand_dims(preprocessed, 0)

        # print(preprocessed_reshape)

        preprocessed_value = sess.run(preprocessed_reshape)
        # scipy.misc.imsave('./PB/crop.jpg', preprocessed_value[0])

        input_tensor = sess.graph.get_tensor_by_name("input:0")

        feature_tensor = sess.graph.get_tensor_by_name(FLAGS.output_node_names + ":0")

        last_feature_tensor = sess.graph.get_tensor_by_name(FLAGS.last_node_names + ":0")

        result = sess.run([feature_tensor], {"input:0": preprocessed_value})
        # last_feature = sess.run([last_feature_tensor],{"input:0": preprocessed_value})
        index = np.argsort(result[0][0])

        #print index[-1]

        #new_image = output_dir + '/' + image_name
        new_image = os.path.join(output_dir, str(index[-1]), image_name)
        new_image_dir = os.path.dirname(new_image)
        if not os.path.exists(new_image_dir):
            os.makedirs(new_image_dir)

        scores = feature_tensor[:, index[-1]]
        last_scores = last_feature_tensor
        grads = K.gradients(scores, last_scores)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([input_tensor], [pooled_grads, last_scores[0]])
        pooled_grads_value, conv_layer_output_value = iterate([preprocessed_value])
        # last_node_names:0层的shape为1280
        for i in range(FLAGS.last_node_size):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]  # 将特征图数组的每个通道乘以这个通道对大象类别重要程度

        heatmap = np.mean(conv_layer_output_value, axis=-1)  # 得到的特征图的逐通道的平均值即为类激活的热力图

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        # plt.matshow(heatmap)
        # plt.show()

        img = cv2.imread(image_path)  # 用cv2加载原始图像

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同

        heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像

        superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子

        cv2.imwrite(new_image, superimposed_img)
        # image = plt.imread('/home/ypf/PycharmProjects/ShowHeatmap/cat.jpg')
        # plt.imshow(image)
        # plt.show()


