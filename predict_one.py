#-*- coding:utf-8 -*-

import tensorflow as tf
import sys

sys.path.append("../")


from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np

import logging

logging.basicConfig(level=logging.WARNING,
                format='%(asctime)s  %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

slim = tf.contrib.slim
'''
CUDA_VISIBLE_DEVICES="0" python predict_batch_new.py \
--net_type=inception_resnet_v2 \
--preprocess_type=inception \
--labels_number=102 \
--image_size=299 \
--check_point=/home/work/wangfei11/data/video_labeling/CKPT/CKPT_InResV2_102_20190419/model.ckpt-192013 \
--image_path=/home/work/wangfei11/TestCase/video_label_testcase/list.txt \
--labels_file=/home/work/wangfei11/data/video_labeling/TF_102_20190419/labels.txt \
--batch_size=24 \
--central_fraction=0


CUDA_VISIBLE_DEVICES="0" python predict_batch_new.py \
--net_type=mobilenet_v2 \
--preprocess_type=inception \
--labels_number=102 \
--image_size=224 \
--check_point=/home/work/wangfei11/data/video_labeling/CKPT/CKPT_MobileNetV2_102_20190419/model.ckpt-299215 \
--image_path=/home/work/wangfei11/TestCase/video_label_testcase/list.txt \
--labels_file=/home/work/wangfei11/data/video_labeling/TF_102_20190419/labels.txt \
--batch_size=50
'''

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('net_type', None, 'model name in nets factory')
tf.app.flags.DEFINE_string('preprocess_type', None, 'preprocess type')
tf.app.flags.DEFINE_integer('labels_number', None, 'input class_num ')
tf.app.flags.DEFINE_integer('image_size', None, 'input image size')
tf.app.flags.DEFINE_string('check_point', None, 'checkpoint file')
tf.app.flags.DEFINE_integer('channel', 3, 'input image channel')
tf.app.flags.DEFINE_string('image_path', None, 'image to predict')
tf.app.flags.DEFINE_string('labels_file', None, 'labels file')
tf.app.flags.DEFINE_bool('quantize', False, 'quantize in PB.')
tf.app.flags.DEFINE_boolean('inres_use_aux', False, 'inception_resnet_v2 use inres aux')
tf.app.flags.DEFINE_float('central_fraction', 0.875, 'Central Crop Fraction.')
tf.app.flags.DEFINE_string('output_node_names', None, 'output_node_names file')


logging.warning('net_type: %s', FLAGS.net_type)
logging.warning('preprocess_type: %s', FLAGS.preprocess_type)
logging.warning('labels_number: %s', FLAGS.labels_number)
logging.warning('image_size: %s', FLAGS.image_size)
logging.warning('check_point: %s', FLAGS.check_point)
logging.warning('channel: %s', FLAGS.channel)
logging.warning('image_path: %s', FLAGS.image_path)
logging.warning('labels_file: %s', FLAGS.labels_file)
logging.warning('quantize: %s', FLAGS.quantize)
logging.warning('inres_use_aux: %s', FLAGS.inres_use_aux)
logging.warning('central_fraction: %s', FLAGS.central_fraction)

if __name__ == "__main__":
    if not FLAGS.net_type:
        raise ValueError('need set model_name')

    if not FLAGS.labels_number:
        raise ValueError('need set labels_num')

    if not FLAGS.preprocess_type:
        raise ValueError('need set preprocess_type')

    if not FLAGS.image_size:
        raise ValueError('need set image_size')
		
    if not FLAGS.check_point:
        raise ValueError('need set check_point')

    if not FLAGS.image_path:
        raise ValueError('need set image_path')

    if not FLAGS.labels_file:
        raise ValueError('need set labels_file')

    sess = tf.Session()

    # 多标签和单标签共用网络结构，只是在prediction的时候使用softmax和sigmoid的区别
    net_type = FLAGS.net_type
    if net_type == "inception_resnet_v2_multilabel":
        net_type = "inception_resnet_v2"

    network_fn = nets_factory.get_network_fn(
        net_type,
        num_classes=FLAGS.labels_number,
        is_training=False)

    image_size = FLAGS.image_size or network_fn.default_image_size
    if not image_size:
        raise ValueError('no default image size, need set model_name')

    input_node = tf.placeholder(tf.float32,
                                shape=(None, image_size, image_size, FLAGS.channel),
                                name="input")

    logits, end_points = network_fn(input_node)

    if FLAGS.quantize:
        tf.contrib.quantize.create_eval_graph()

    # 定义模型输入
    if FLAGS.net_type == 'inception_resnet_v2' or FLAGS.net_type == 'inception_resnet_v2_dropblock':
        output_node_names = "InceptionResnetV2/Logits/Predictions"
        if FLAGS.inres_use_aux:
            print("inres use aux")
            new_logit = (end_points['AuxLogits'] + end_points['Logits']) / 2.0
            inres_output = tf.nn.softmax(new_logit, name='InceptionResnetV2/Logits/Predictions_Mix_Aux')
            #output_node_names = "InceptionResnetV2/Logits/Predictions_Mix_Aux,InceptionResnetV2/Logits/Predictions"
            output_node_names = "InceptionResnetV2/Logits/Predictions_Mix_Aux"
    elif FLAGS.net_type == 'inception_resnet_v2_multilabel':
        new_logit = end_points['AuxLogits']
        inres_output = tf.nn.sigmoid(new_logit, name='InceptionResnetV2/Logits/Predictions_multilabel')
        #output_node_names = "InceptionResnetV2/Logits/Predictions_multilabel,InceptionResnetV2/Logits/Predictions"
        output_node_names = "InceptionResnetV2/Logits/Predictions_multilabel"
        if FLAGS.inres_use_aux:
            print("inres use aux")
            new_logit = (end_points['AuxLogits'] + end_points['Logits']) / 2.0
            inres_output_mix = tf.nn.sigmoid(new_logit, name='InceptionResnetV2/Logits/Predictions_Mix_Aux')
            #output_node_names = "InceptionResnetV2/Logits/Predictions_Mix_Aux,InceptionResnetV2/Logits/Predictions_multilabel,InceptionResnetV2/Logits/Predictions"
            output_node_names = "InceptionResnetV2/Logits/Predictions_Mix_Aux"
    elif FLAGS.net_type == 'inception_v3':
        output_node_names = "InceptionV3/Predictions/Softmax"
    elif FLAGS.net_type == 'resnet_v2_101':
        output_node_names = "resnet_v2_101/predictions/Softmax"
    elif FLAGS.net_type == 'resnet_v2_152':
        output_node_names = "resnet_v2_152/predictions/Softmax"
    elif FLAGS.net_type == 'inception_v4':
        output_node_names = "InceptionV4/Logits/Predictions"
    elif FLAGS.net_type == 'nasnet_large':
        output_node_names = "final_layer/predictions"
    elif FLAGS.net_type == 'mobilenet_v2':
        output_node_names = "MobilenetV2/Predictions/Reshape_1"
    elif FLAGS.net_type == 'mobilenet_v2_multilabel':
        output_node_names = "MobilenetV2/Predictions"
    elif FLAGS.net_type == 'mobilenet_v1_qc':
        output_node_names = "MobilenetV1/Predictions_softmax"
    elif FLAGS.net_type == 'pelee_net_multilabel':
        output_node_names = "pelee_net/Logits/Predictions"
    elif FLAGS.net_type == 'shufflenet_multilabel':
        output_node_names = "ShuffleNet/Predictions"
    elif FLAGS.net_type == 'shufflenet_v2_multilabel':
        output_node_names = "Shufflenet_v2/Predictions"
    elif FLAGS.net_type == 'shufflenet_v2d15_multilabel':
        output_node_names = "Shufflenet_v2/Predictions"
    elif FLAGS.net_type == 'shufflenet_v2':
        output_node_names = "Shufflenet_v2/Predictions"
    elif FLAGS.net_type == 'mobilenet_v3_large':
        output_node_names = "MobilenetV3_large/Logits/Predictions_softmax"
    elif FLAGS.output_node_names:
        output_node_names = FLAGS.output_node_names
    else:
        raise ValueError('net type not support')

    if FLAGS.output_node_names:
        output_node_names = FLAGS.output_node_names

    print("output_node_names: %s" % output_node_names)

    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # 指定具体的checkpoint文件
    input_checkpoint_path = FLAGS.check_point
    saver.restore(sess, input_checkpoint_path)  # 加载checkpoint的参数到图模型

    lines = open(FLAGS.labels_file, encoding='utf-8').read().splitlines()
    id_2_label = {}
    for line in lines:
        sep = line.split(":")
        if len(sep) == 2:
            id_2_label[int(sep[0])] = sep[1]

    IMG_RESIZE = FLAGS.image_size

    input_node_names = 'input:0'
    feature_tensor = sess.graph.get_tensor_by_name(output_node_names + ":0")
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(FLAGS.preprocess_type, is_training=False)

    ph_files = tf.placeholder(tf.string, shape=[1], name='ph_files')

    image_input_tensor = tf.image.decode_jpeg(ph_files[0])
    image_input_reshape_tensor = image_preprocessing_fn(image_input_tensor, IMG_RESIZE, IMG_RESIZE, 
                                                        central_fraction = FLAGS.central_fraction)
    image_input_reshape_tensor = tf.reshape(image_input_reshape_tensor, [1, IMG_RESIZE, IMG_RESIZE, 3])
    input_tensor = image_input_reshape_tensor

    try:
        try:
            image_data = tf.gfile.FastGFile(FLAGS.image_path, 'rb').read()
        except Exception as e:
            print("tf.gfile.FastGFile Exception!", e)

        logging.warning('processing image  %s', FLAGS.image_path)

        image_input = sess.run(input_tensor, {'ph_files:0': [image_data]})
        features = sess.run(feature_tensor, {input_node_names: image_input})
        predictions = features[0]
        sort_idx = predictions.argsort()[::-1]
        for idx in sort_idx:
            curr_result = str(predictions[idx]) + "\t" + id_2_label[idx]
            print('{}'.format(curr_result))

    except Exception as e:
        print(Exception, ":", e)
        L_input_image_list = []
        curr_batch_file = []
