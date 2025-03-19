import tensorflow as tf
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ckpt_path = sys.argv[1]

with tf.Session() as sess:
    for var_name, _ in tf.contrib.framework.list_variables(ckpt_path):
        print(var_name)
