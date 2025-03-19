# -*- coding: utf-8 -*-  
import os
import cv2
import ipdb
import sys
import numpy as np
import tensorflow as tf
import logging
import time
from tqdm import tqdm
from tensorflow.python.platform import gfile
sys.path.append("../")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

_R_MEAN = 0.485
_G_MEAN = 0.456
_B_MEAN = 0.406
_R_STD = 0.229
_G_STD = 0.224
_B_STD = 0.225
ARRAY_36 = np.asarray([15,17,18,25,28,32,8,16,24,23,29,35,1,6,7,20,26,31,5,9,10,11,0,4,12,13,22,33,3,14,34,21,27,30,19,2])
ARRAY_34 = np.asarray([14,16,17,23,26,30,7,15,22,21,27,33,1,5,6,18,24,29,4,8,9,10,0,3,11,12,20,31,2,13,32,19,25,28])





sys.path.append("../")
slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('image_gt_list', './imListTest_gt_34.txt', 'image gt file')
tf.app.flags.DEFINE_string('pb_path', "./model_0531.pb", 'pb file')
tf.app.flags.DEFINE_string('output_node_names', 'Squeeze_1', 'output_node_names')
tf.app.flags.DEFINE_string('result_file', "model_0531.txt", 'result_file')
tf.app.flags.DEFINE_string('img_root_path', "/", 'img_root_path')
tf.app.flags.DEFINE_integer('img_size', 224, 'img_size')

# tf.app.flags.DEFINE_string('preprocess_name', "inception", 'preprocess_name')

def _aspect_changing_resize(image, resize_side):
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [resize_side, resize_side],
                                           align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image

def _image_normalization(image, means, stds):
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] = (channels[i] - means[i]) / stds[i]
    return tf.concat(axis=2, values=channels)

def preprocess_for_eval_pb(image, height, width,scope=None):
    # with tf.name_scope(scope, 'eval_image', [image, height, width]):
    image = _aspect_changing_resize(image, height)
    image = image/255.0
    image.set_shape([height, width, 3])
    image = tf.to_float(image)
    if height and width:
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
        image = tf.squeeze(image, [0])

    return _image_normalization(image, [_R_MEAN, _G_MEAN, _B_MEAN],
                                [_R_STD, _G_STD, _B_STD])



if __name__ == "__main__":
    if not FLAGS.pb_path:
        raise ValueError('need set pb_path')

    if not FLAGS.output_node_names:
        raise ValueError('need set output_node_names')

    if not FLAGS.image_gt_list:
        raise ValueError('need set image_gt_list')

    if not FLAGS.img_root_path:
        raise ValueError('need set img_root_path')

    img_root_path = FLAGS.img_root_path
    
    graph_def = tf.GraphDef()
    with open(FLAGS.pb_path) as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    sess = tf.Session()

    img_gt_lines = open(FLAGS.image_gt_list).read().splitlines()
    test_img_list = [line.split()[1] for line in img_gt_lines]
    logging.warning('len(test_img_list) %s', len(test_img_list))

    result_file = FLAGS.result_file

    IMG_RESIZE = FLAGS.img_size
    logging.warning('result_file: %s', result_file)
    logging.warning('IMG_RESIZE: %s', IMG_RESIZE)

    BATCH_COUNT = 1
    startIdx = 0
    endIdx = len(test_img_list)
    input_node_names = 'input:0'
    feature_tensor = sess.graph.get_tensor_by_name(FLAGS.output_node_names + ":0")
    input_tensor = sess.graph.get_tensor_by_name(input_node_names)
    print(input_tensor)

    ph_files = tf.placeholder(tf.string, shape=[BATCH_COUNT], name='ph_files')
    image_list = []
    for i in range(BATCH_COUNT):
        image_input_tensor = tf.image.decode_jpeg(ph_files[i])
        image_input_reshape_tensor = preprocess_for_eval_pb(image_input_tensor, IMG_RESIZE, IMG_RESIZE)
        image_input_reshape_tensor = tf.reshape(image_input_reshape_tensor, [1, IMG_RESIZE, IMG_RESIZE, 3])
        image_input_reshape_tensor = tf.transpose(image_input_reshape_tensor, [0, 3, 1, 2])
        image_list.append(image_input_reshape_tensor)
    input_tensor = tf.concat(image_list, 0)

    L_file = []
    L_result = []
    L_input_image_list = []
    curr_batch_file = []
    #time1 = time.time()
    #endIdx
    for i in tqdm(range(startIdx, endIdx)):
        try:
            image_path = os.path.join(img_root_path, test_img_list[i])
            try:
                image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            except Exception, e:
                print("tf.gfile.FastGFile Exception!", e, image_path)
                continue

            # if (i + 1 - startIdx) % BATCH_COUNT == 0:
            #     logging.warning('%s[%d-%d]  processing image  %d  %s', FLAGS.image_gt_list, startIdx, \
            #                     endIdx, i, image_path)
            # print(FLAGS.image_gt_list, startIdx, endIdx, i, image_path)

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
		    #time2 = time.time()
		    #print(time2-time1,"time2-time1")
            image_input = sess.run(input_tensor, {'ph_files:0': L_input_image_list})
            #time3 = time.time()
		    #print(time3-time2,"time3-time2")
            features = sess.run(feature_tensor, {input_node_names: image_input})
            for k in range(curr_batch_image_count):
		    # time4 = time.time()
		    # print(time4-time3,"time4-time3")
                predictions = features[k]
                sigmoid_predictions = 1.0 / (1.0 + np.exp(-predictions))
		    #time5 = time.time()
		    # print(time5-time4,"time5-time4")
                    # top_k = predictions.argsort()[-5:][::-1]
                    # result = [id_2_label[node_id] for node_id in top_k]
                    # result_str = " ".join(result)
                    # ipdb.set_trace()
                predictions_adj = sigmoid_predictions[ARRAY_34]
                pridictions_str = [str(item) for item in predictions_adj]
                L_result.append("\t".join(pridictions_str) + "\n")
            L_file.extend(curr_batch_file)
            L_input_image_list = []
            curr_batch_file = []
		#time1 = time.time()
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









