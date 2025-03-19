#-*- coding:utf-8 -*-

import tensorflow as tf
import sys

sys.path.append("../")


from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np

import logging

PY3FLAG = sys.version_info > (3,0)
if PY3FLAG:
    open_fn = open
else:
    import codecs
    open_fn = codecs.open

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
--image_list=/home/work/wangfei11/TestCase/video_label_testcase/list.txt \
--labels_file=/home/work/wangfei11/data/video_labeling/TF_102_20190419/labels.txt \
--batch_size=24 \
--central_fraction=0


CUDA_VISIBLE_DEVICES="0" python predict_batch_new.py \
--net_type=mobilenet_v2 \
--preprocess_type=inception \
--labels_number=102 \
--image_size=224 \
--check_point=/home/work/wangfei11/data/video_labeling/CKPT/CKPT_MobileNetV2_102_20190419/model.ckpt-299215 \
--image_list=/home/work/wangfei11/TestCase/video_label_testcase/list.txt \
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
tf.app.flags.DEFINE_string('image_list', None, 'image to predict')
tf.app.flags.DEFINE_string('labels_file', None, 'labels file')
tf.app.flags.DEFINE_integer('batch_size', 24, 'batch size')
tf.app.flags.DEFINE_bool('quantize', False, 'quantize in PB.')
tf.app.flags.DEFINE_boolean('inres_use_aux', False, 'inception_resnet_v2 use inres aux')
tf.app.flags.DEFINE_float('central_fraction', 0.875, 'Central Crop Fraction.')
tf.app.flags.DEFINE_boolean('multi_label_output', False, 'multi_label_output')
tf.app.flags.DEFINE_float('multi_label_output_threshold', 0.1, 'multi_label_output_threshold.')
tf.app.flags.DEFINE_string('output_node_names', None, 'output_node_names file')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_bool('gen_score_file', False, 'gen_score_file.')


logging.warning('net_type: %s', FLAGS.net_type)
logging.warning('preprocess_type: %s', FLAGS.preprocess_type)
logging.warning('labels_number: %s', FLAGS.labels_number)
logging.warning('image_size: %s', FLAGS.image_size)
logging.warning('check_point: %s', FLAGS.check_point)
logging.warning('channel: %s', FLAGS.channel)
logging.warning('image_list: %s', FLAGS.image_list)
logging.warning('labels_file: %s', FLAGS.labels_file)
logging.warning('batch_size: %s', FLAGS.batch_size)
logging.warning('quantize: %s', FLAGS.quantize)
logging.warning('inres_use_aux: %s', FLAGS.inres_use_aux)
logging.warning('central_fraction: %s', FLAGS.central_fraction)
logging.warning('moving_average_decay: {}'.format(FLAGS.moving_average_decay))
logging.warning('multi_label_output: %s', FLAGS.multi_label_output)
logging.warning('gen_score_file: %s', FLAGS.gen_score_file)

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

    if not FLAGS.image_list:
        raise ValueError('need set image_list')

    if not FLAGS.labels_file:
        raise ValueError('need set labels_file')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

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
        output_node_names = "MobilenetV1/Predictions"
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

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
    else:
      variables_to_restore = slim.get_variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)

    # 指定具体的checkpoint文件
    input_checkpoint_path = FLAGS.check_point
    saver.restore(sess, input_checkpoint_path)  # 加载checkpoint的参数到图模型

    '''
    with open("/tmp/opTotal.log", "w") as f:
        for op in sess.graph.get_operations():
            f.write(op.name + "\n")
    '''


    test_img_list = open_fn(FLAGS.image_list, encoding='utf-8').read().splitlines()
    # 对于有些测试集，第一列是路径，第二列和以后是标签，这里只取第一列
    test_img_list = [item.split()[0] for item in test_img_list]
    logging.warning('len(test_img_list) %s', len(test_img_list))

    lines = open_fn(FLAGS.labels_file, encoding='utf-8').read().splitlines()
    id_2_label = {}
    for line in lines:
        sep = line.split(":")
        if len(sep) == 2:
            id_2_label[int(sep[0])] = sep[1]

    BATCH_COUNT = FLAGS.batch_size
    IMG_RESIZE = FLAGS.image_size
    startIdx = 0
    endIdx = len(test_img_list)
    input_node_names = 'input:0'
    feature_tensor = sess.graph.get_tensor_by_name(output_node_names + ":0")
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(FLAGS.preprocess_type, is_training=False)

    ph_files = tf.placeholder(tf.string, shape=[BATCH_COUNT], name='ph_files')
    image_list = []
    for i in range(BATCH_COUNT):
        image_input_tensor = tf.image.decode_jpeg(ph_files[i])
        image_input_reshape_tensor = image_preprocessing_fn(image_input_tensor, IMG_RESIZE, IMG_RESIZE, 
                                                            central_fraction = FLAGS.central_fraction)
        image_input_reshape_tensor = tf.reshape(image_input_reshape_tensor, [1, IMG_RESIZE, IMG_RESIZE, 3])
        image_list.append(image_input_reshape_tensor)
    input_tensor = tf.concat(image_list, 0)

    L_file = []
    L_result = []
    L_scores = []
    L_input_image_list = []
    curr_batch_file = []
    for i in range(startIdx, endIdx):
        try:
            image_path = test_img_list[i]

            try:
                image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            except Exception as e:
                print("tf.gfile.FastGFile Exception!", e)
                continue

            if (i + 1 - startIdx) % BATCH_COUNT == 1:
                logging.warning('%s[%d-%d]  processing image  %d  %s', FLAGS.image_list, startIdx, endIdx, i, image_path)

            L_input_image_list.append(image_data)
            # 先把文件名记录下来
            #sep = test_img_list[i].split('/')
            #curr_batch_file.append(sep[len(sep) - 1] + '\t')
            curr_batch_file.append(test_img_list[i] + '\t')

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
                    curr_result = str(predictions[top_k[0]]) + "\t" + id_2_label[top_k[0]]
                    if FLAGS.multi_label_output:
                        for kk in range(1, 5):
                            if predictions[top_k[kk]] > FLAGS.multi_label_output_threshold:
                                curr_result += "\t" + str(predictions[top_k[kk]]) + "\t" + id_2_label[top_k[kk]]
                    L_result.append(curr_result + "\n")
                    
                    pridictions_str = [str(item) for item in predictions]
                    L_scores.append("\t".join(pridictions_str) + "\n")

                L_file.extend(curr_batch_file)
                L_input_image_list = []
                curr_batch_file = []

        except Exception as e:
            print(Exception, ":", e)
            L_input_image_list = []
            curr_batch_file = []

    if len(L_file) != len(L_result):
        logging.warning('count NOT Equal!')
        exit(1)

    out_lines = []
    for i in range(len(L_file)):
        out_lines.append(L_file[i] + L_result[i])

    open_fn('result_img_prediction.txt', 'w', encoding='utf-8').writelines(out_lines)
    
    if FLAGS.gen_score_file:
        out_lines = []
        for i in range(len(L_file)):
            out_lines.append(L_file[i] + L_scores[i])
        open_fn('result_score.txt', 'w', encoding='utf-8').writelines(out_lines)
