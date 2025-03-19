#-*- coding:utf-8 -*-

import tensorflow as tf
import sys

sys.path.append("../")

from preprocessing import preprocessing_factory
import numpy as np

import logging

logging.basicConfig(level=logging.WARNING,
				format='%(asctime)s  %(message)s',
				datefmt='%Y-%m-%d %H:%M:%S')

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('image_list', None, 'image to predict')
tf.app.flags.DEFINE_string('pb_path', None, 'pb file')
tf.app.flags.DEFINE_string('labels_file', None, 'labels file')
tf.app.flags.DEFINE_string('output_node_names', None, 'output_node_names')
tf.app.flags.DEFINE_integer('no_crop', 1, 'no_crop')
tf.app.flags.DEFINE_integer('batch_size', 24, 'batch size')
tf.app.flags.DEFINE_integer('image_size', None, 'input image size')
tf.app.flags.DEFINE_string('preprocess_type', None, 'preprocess type')
tf.app.flags.DEFINE_integer('mul_label', 0, 'mul_label')
tf.app.flags.DEFINE_string('result_file', None, 'result_file')

# tf.app.flags.DEFINE_integer(
#	 'batch',
#	 1,
#	 'labels number'
# )

'''
CUDA_VISIBLE_DEVICES="0" \
python predict_batch_pb.py \
--pb_path=/home/work/wangfei11/data/AI_Label_Poem/mnasnet_a1_208_3_20200116.pb \
--output_node_names=Mnasnet/Predictions_sigmoid \
--preprocess_type=inception --image_size=224 --batch_size=100 \
--labels_file=/home/work/wangfei11/data/AI_Label_Poem/labels_208.txt \
--image_list=/home/work/wangfei11/data/AI_Label_Poem/list_1125.txt \
--result_file=/home/work/wangfei11/data/AI_Label_Poem/result_img_prediction_1125.txt \
--mul_label=1

CUDA_VISIBLE_DEVICES="0" \
python predict_batch_pb.py \
--pb_path=/home/work/wangfei11/data/AI_Label_Poem/scene_43_mv1qc_20200120_40147_quant.pb \
--output_node_names=MobilenetV1/Predictions_sigmoid \
--preprocess_type=scene_new --image_size=224 --batch_size=100 \
--labels_file=/home/work/wangfei11/data/AI_Label_Poem/labels_43.txt \
--image_list=/home/work/wangfei11/data/AI_Label_Poem/list_detection_image.txt \
--result_file=/home/work/wangfei11/data/AI_Label_Poem/result_img_prediction_tmp.txt \
--mul_label=1
'''

if __name__ == "__main__":
	if not FLAGS.pb_path:
		raise ValueError('need set pb_path')

	if not FLAGS.image_list:
		raise ValueError('need set image_list')

	if not FLAGS.labels_file:
		raise ValueError('need set labels_file')

	if not FLAGS.output_node_names:
		raise ValueError('need set output_node_names')

	if not FLAGS.image_size:
		raise ValueError('need set image_size')

	if not FLAGS.preprocess_type:
		raise ValueError('need set preprocess_type')

	graph_def = tf.GraphDef()
	with open(FLAGS.pb_path) as f:
		graph_def.ParseFromString(f.read())
	tf.import_graph_def(graph_def, name='')

	sess = tf.Session()

	test_img_list = open(FLAGS.image_list).read().splitlines()
	logging.warning('len(test_img_list) %s', len(test_img_list))

	lines = open(FLAGS.labels_file).read().splitlines()
	id_2_label = {}
	for line in lines:
		sep = line.split(":")
		if len(sep) == 2:
			id_2_label[int(sep[0])] = sep[1]

	result_file = 'result_img_prediction.txt'
	central_fraction = 0.875
	if FLAGS.no_crop == 1:
		result_file = 'result_img_prediction_no_crop.txt'
		central_fraction = None
	if FLAGS.result_file:
		result_file = FLAGS.result_file
	logging.warning('result_file: %s', result_file)
	logging.warning('central_fraction: %s', central_fraction)

	BATCH_COUNT = FLAGS.batch_size
	IMG_RESIZE = FLAGS.image_size
	startIdx = 0
	endIdx = len(test_img_list)
	input_node_names = 'input:0'
	feature_tensor = sess.graph.get_tensor_by_name(FLAGS.output_node_names + ":0")
	image_preprocessing_fn = preprocessing_factory.get_preprocessing(FLAGS.preprocess_type, is_training=False)

	ph_files = tf.placeholder(tf.string, shape=[BATCH_COUNT], name='ph_files')
	image_list = []
	for i in range(BATCH_COUNT):
		image_input_tensor = tf.image.decode_jpeg(ph_files[i])
		image_input_reshape_tensor = image_preprocessing_fn(image_input_tensor, IMG_RESIZE, IMG_RESIZE, central_fraction=central_fraction)
		image_input_reshape_tensor = tf.reshape(image_input_reshape_tensor, [1, IMG_RESIZE, IMG_RESIZE, 3])
		image_list.append(image_input_reshape_tensor)
	input_tensor = tf.concat(image_list, 0)

	L_file = []
	L_result = []
	L_input_image_list = []
	curr_batch_file = []
	for i in range(startIdx, endIdx):
		try:
			image_path = test_img_list[i]

			try:
				image_data = tf.gfile.FastGFile(image_path, 'rb').read()
			except Exception, e:
				print("tf.gfile.FastGFile Exception!", e)
				continue

			if (i + 1 - startIdx) % BATCH_COUNT == 1:
				logging.warning('%s[%d-%d]  processing image  %d  %s', FLAGS.image_list, startIdx, endIdx, i, image_path)

			L_input_image_list.append(image_data)
			# 先把文件名记录下来
			#sep = test_img_list[i].split('/')
			#curr_batch_file.append(sep[len(sep) - 1] + '\t')
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
					top_k = predictions.argsort()[-5:][::-1]
					#result = [id_2_label[node_id] for node_id in top_k]
					#result_str = " ".join(result)
					curr_result = ""
					if FLAGS.mul_label == 0:
						curr_result = "\t" + str(predictions[top_k[0]]) + "\t" + id_2_label[top_k[0]]
					else:
						for kk in range(5):
							if predictions[top_k[kk]] < 0.1:
								break
							curr_result = curr_result + "\t" + str(predictions[top_k[kk]]) + "\t" + id_2_label[top_k[kk]]
					L_result.append(curr_result)
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
		out_lines.append(L_file[i] + L_result[i] + "\n")

	open(result_file, 'w').writelines(out_lines)
