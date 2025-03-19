#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import tensorflow as tf
import cv2
import numpy as np
import time
import os
import sys
import fnmatch
from stat import *

def load_graph(model_path):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    graph_def = None
    with tf.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.GraphDef.FromString(f.read())

    if graph_def is None:
        raise RuntimeError('Cannot find inference graph in tar archive.')

    # Then, we can use again a convenient built-in function to import a
    # graph_def into the current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    return graph

'''
CUDA_VISIBLE_DEVICES="6"  python test_one_img.py PBs/frozen_graph_96051_34cls_ft_0117_2.pb BadCase/food.jpg
'''

if len(sys.argv) < 3:
    print("Usage: %s pb_path img_path" % sys.argv[0])
    exit()
    
#graph = load_graph('./frozen_graph_19458_qc_ft.pb')
graph = load_graph(sys.argv[1])
img_path = sys.argv[2]
num_classes = 34


#x = graph.get_tensor_by_name('input:0')
#y = graph.get_tensor_by_name('MobilenetV1/Predictions:0')
INPUT_TENSOR_NAME = 'input:0'
OUTPUT_TENSOR_NAME = 'MobilenetV1/Predictions:0'
WIDTH, HEIGHT = 224, 224
BATCH_SIZE = 50

pop_preprocess = True
if pop_preprocess:
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]
else:
    mean_rgb = [127.5, 127.5, 127.5]
    std_rgb = [127.5, 127.5, 127.5]
    
label_lines = open("label_34.txt").read().splitlines()
id_2_label = {int(line.split()[1]): line.split()[0] for line in label_lines}
#print(id_2_label)

#fout = open('scores_19458_qc_ft.txt', 'w')
start = time.time()
with tf.Session(graph=graph) as sess:
    im = cv2.imread(img_path)
    im = cv2.resize(im, (WIDTH, HEIGHT))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.array(im, dtype=np.float32)
    # im /= 255.0
    # if not use_mixup:
    #     im = 2.0 * (im - 0.5)
    # else:
    #     im -= [0.485, 0.456, 0.406]
    #     im /= [0.229, 0.224, 0.225]
    im = (im - mean_rgb) / std_rgb
    im = np.expand_dims(im, axis=0)

    #y_out = sess.run(y, feed_dict={ x: im })
    y_out = sess.run(OUTPUT_TENSOR_NAME, feed_dict={INPUT_TENSOR_NAME: im})
    
    scores = np.squeeze(y_out)
    print scores.argsort()[-5:][::-1]
    print(img_path)
    for i in range(len(scores)):
        print('\t{}'.format(scores[i]))
    for idx in scores.argsort()[-5:][::-1]:
        print("%s : %f" % (id_2_label[idx], scores[idx]))

print 'Computation time: {0}'.format(time.time() - start)
