#-*- coding:utf-8 -*-

"""
Created on 2019-03-19

@author: wangfei
"""

import tensorflow as tf
import sys

sys.path.append("../")

from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np
import os
import logging

logging.basicConfig(level=logging.WARNING,
                format='%(asctime)s  %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('image_gt_list', None, 'image gt file')
tf.app.flags.DEFINE_string('net_type', None, 'model name in nets factory')
tf.app.flags.DEFINE_integer('labels_number', None, 'input class_num ')
tf.app.flags.DEFINE_string('check_point', None, 'checkpoint file')
tf.app.flags.DEFINE_string('output_node_names', None, 'output_node_names')
tf.app.flags.DEFINE_integer('no_crop', 1, 'no_crop')
tf.app.flags.DEFINE_integer('img_size', 224, 'img_size')
tf.app.flags.DEFINE_string('result_file', "result_file.txt", 'result_file')
tf.app.flags.DEFINE_string('img_root_path', None, 'img_root_path')
tf.app.flags.DEFINE_string('preprocess_name', "inception", 'preprocess_name')
tf.app.flags.DEFINE_integer('channel', 3, 'input image channel')

# tf.app.flags.DEFINE_integer(
#     'batch',
#     1,
#     'labels number'
# )

'''
CUDA_VISIBLE_DEVICES="4"   python compute_scores_byCKPT.py \
                --net_type=efficient_net_b0 \
                --labels_number=34 \
                --img_size=224 \
                --check_point=/home/work/wangfei11/TFRec_LMDB_CKPT/20190311_34/ckpt_tf_efficientnetb0_fromscratch/model.ckpt-17010 \
                --output_node_names=efficientnet-b0/Logits/Predictions_sigmoid \
                --image_gt_list=/home/work/wangfei11/TestCase/new_testimg_bycls_size_suffix/imListTest_gt_34.txt \
                --img_root_path=/ \
                --preprocess_name=scene \
                --result_file=/home/work/wangfei11/TFRec_LMDB_CKPT/20190311_34/ckpt_tf_efficientnetb0_fromscratch/result_score_17010.txt
'''



if __name__ == "__main__":
    if not FLAGS.net_type:
        raise ValueError('need set model_name')

    if not FLAGS.labels_number:
        raise ValueError('need set labels_num')
		
    if not FLAGS.check_point:
        raise ValueError('need set check_point')

    if not FLAGS.output_node_names:
        raise ValueError('need set output_node_names')

    if not FLAGS.image_gt_list:
        raise ValueError('need set image_gt_list')

    if not FLAGS.img_root_path:
        raise ValueError('need set img_root_path')

    img_root_path = FLAGS.img_root_path
    

    sess = tf.Session()

    # 多标签和单标签共用网络结构，只是在prediction的时候使用softmax和sigmoid的区别
    net_type = FLAGS.net_type
    if net_type == "inception_resnet_v2_multilabel":
        net_type = "inception_resnet_v2"

    network_fn = nets_factory.get_network_fn(
        net_type,
        num_classes=FLAGS.labels_number,
        is_training=False)

    image_size = FLAGS.img_size or network_fn.default_image_size
    if not image_size:
        raise ValueError('no default image size, need set model_name')

    input_node = tf.placeholder(tf.float32,
                                shape=(None, image_size, image_size, FLAGS.channel),
                                name="input")

    logits, end_points = network_fn(input_node)

    print("output_node_names: %s" % FLAGS.output_node_names)

    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # 指定具体的checkpoint文件
    input_checkpoint_path = FLAGS.check_point
    saver.restore(sess, input_checkpoint_path)  # 加载checkpoint的参数到图模型

    img_gt_lines = open(FLAGS.image_gt_list).read().splitlines()
    test_img_list = [line.split()[1] for line in img_gt_lines]
    logging.warning('len(test_img_list) %s', len(test_img_list))

    result_file = FLAGS.result_file
    central_fraction = 0.875
    if FLAGS.no_crop == 1:
        central_fraction = None
    IMG_RESIZE = FLAGS.img_size
    preprocess_name = FLAGS.preprocess_name
    logging.warning('result_file: %s', result_file)
    logging.warning('central_fraction: %s', central_fraction)
    logging.warning('IMG_RESIZE: %s', IMG_RESIZE)
    logging.warning('preprocess_name: %s', preprocess_name)

    BATCH_COUNT = 50
    startIdx = 0
    endIdx = len(test_img_list)
    input_node_names = 'input:0'
    feature_tensor = sess.graph.get_tensor_by_name(FLAGS.output_node_names + ":0")
    input_tensor = sess.graph.get_tensor_by_name(input_node_names)
    print(input_tensor)
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocess_name, is_training=False)

    ph_files = tf.placeholder(tf.string, shape=[BATCH_COUNT], name='ph_files')
    image_list = []
    for i in range(BATCH_COUNT):
        image_input_tensor = tf.image.decode_jpeg(ph_files[i])
        #image_input_reshape_tensor = image_preprocessing_fn(image_input_tensor, IMG_RESIZE, IMG_RESIZE, \
        #                                                    central_fraction=central_fraction)
        image_input_reshape_tensor = image_preprocessing_fn(image_input_tensor, IMG_RESIZE, IMG_RESIZE)
        image_input_reshape_tensor = tf.reshape(image_input_reshape_tensor, [1, IMG_RESIZE, IMG_RESIZE, 3])
        image_list.append(image_input_reshape_tensor)
    input_tensor = tf.concat(image_list, 0)

    L_file = []
    L_result = []
    L_input_image_list = []
    curr_batch_file = []
    for i in range(startIdx, endIdx):
        try:
            image_path = os.path.join(img_root_path, test_img_list[i])

            try:
                image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            except Exception, e:
                print("tf.gfile.FastGFile Exception!", e, image_path)
                continue

            if (i + 1 - startIdx) % BATCH_COUNT == 1:
                logging.warning('%s[%d-%d]  processing image  %d  %s', FLAGS.image_gt_list, startIdx, \
                                endIdx, i, image_path)

            L_input_image_list.append(image_data)
            # 先把文件名记录下来
            #sep = test_img_list[i].split('/')
            #curr_batch_file.append('\t'.join(sep[len(sep) - 2:]) + '\t')
            curr_batch_file.append(test_img_list[i])

            # 在填满一个batch，或者所有图片处理完后，批量计算一次
            if len(L_input_image_list) % BATCH_COUNT == 0 or i == endIdx - 1 or i >= len(test_img_list) - 1:
                # 如果最后一波不是正好是BATCH_COUNT个，那么补齐
                curr_batch_image_count = len(L_input_image_list)
                # logging.warning('i:%d curr_batch_image_count:%d', i, curr_batch_image_count)
                if curr_batch_image_count % BATCH_COUNT != 0:
                    for _ in range(curr_batch_image_count, BATCH_COUNT):
                        L_input_image_list.append(image_data)
                image_input = sess.run(input_tensor, {'ph_files:0': L_input_image_list})
                features = sess.run(feature_tensor, {input_node_names: image_input})
                for k in range(curr_batch_image_count):
                    predictions = features[k]
                    #top_k = predictions.argsort()[-5:][::-1]
                    #result = [id_2_label[node_id] for node_id in top_k]
                    #result_str = " ".join(result)
                    pridictions_str = [str(item) for item in predictions]
                    L_result.append("\t".join(pridictions_str) + "\n")
                L_file.extend(curr_batch_file)
                L_input_image_list = []
                curr_batch_file = []

        except Exception, e:
            print(Exception, ":", e)
            L_input_image_list = []
            curr_batch_file = []

    if len(L_file) != len(L_result):
        logging.warning('count NOT Equal!')
        exit(1)

    out_lines = []
    for i in range(len(L_file)):
        out_lines.append(L_file[i] + "\t" + L_result[i])

    open(result_file, 'w').writelines(out_lines)
