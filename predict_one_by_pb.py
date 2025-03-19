#-*- coding:utf-8 -*-

import tensorflow as tf
import sys

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
tf.app.flags.DEFINE_string('image_path', None, 'image to predict')
tf.app.flags.DEFINE_string('pb_path', None, 'pb file')
tf.app.flags.DEFINE_string('labels_file', None, 'labels file')
tf.app.flags.DEFINE_string('output_node_names', None, 'output_node_names')
tf.app.flags.DEFINE_integer('image_size', 299, 'image_size.')
tf.app.flags.DEFINE_string('preprocess_type', None, 'preprocess type')


'''
python predict_by_pb.py --pb_path=/data_old/ffwang/PBs/Combine.pb \
--image_path=/data_old/ffwang/PBs/meiyuan.jpg \
--output_node_names=predictions \
--labels_file=/data_old/ffwang/PBs/labels_20181018_1500.txt


python predict_by_pb.py --pb_path=/data_old/ffwang/PBs/imgscene_batchNone_20181018_1500_59933.pb \
--image_path=/data_old/ffwang/PBs/meiyuan.jpg \
--output_node_names=InceptionResnetV2/Logits/Predictions \
--labels_file=/data_old/ffwang/PBs/labels_20181018_1500.txt

python predict_one_by_pb.py --pb_path=d:/OnlineModel/Album_tag/20200119_videolabal_inoutdoor/mnasnet_a1_208_3_20200116.pb --image_path=testid.jpg --output_node_names=Mnasnet/Predictions_sigmoid --preprocess_type=inception --image_size=224 --labels_file=d:/OnlineModel/Album_tag/20200119_videolabal_inoutdoor/labels_208.txt
'''

if __name__ == "__main__":
    if not FLAGS.pb_path:
        raise ValueError('need set pb_path')

    if not FLAGS.image_path:
        raise ValueError('need set image_path')

    if not FLAGS.labels_file:
        raise ValueError('need set labels_file')

    if not FLAGS.output_node_names:
        raise ValueError('need set output_node_names')

    if not FLAGS.preprocess_type:
        raise ValueError('need set preprocess_type')

    graph_def = tf.GraphDef()
    with open(FLAGS.pb_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    sess = tf.Session()

    image_preprocessing_fn = preprocessing_factory.get_preprocessing(FLAGS.preprocess_type, is_training=False)
    #image_data = tf.gfile.FastGFile(FLAGS.image_path, 'rb').read()
    with open(FLAGS.image_path, 'rb') as f:
        image_data = f.read()
    image_data_tensor = tf.image.decode_jpeg(image_data)
    preprocessed = image_preprocessing_fn(image_data_tensor, FLAGS.image_size, FLAGS.image_size)
    preprocessed_reshape = tf.expand_dims(preprocessed, 0)
    print(preprocessed_reshape)

    preprocessed_value = sess.run(preprocessed_reshape)
    #scipy.misc.imsave('./PB/crop.jpg', preprocessed_value[0])

    feature_tensor = sess.graph.get_tensor_by_name(FLAGS.output_node_names + ":0")
    features = sess.run([feature_tensor], {"input:0": preprocessed_value})
    scores=features[0][0]
    #before squeeze use this 
    #scores=features[0][0][0][0]
    print(scores)
    index = np.argsort(scores)
    #print index

    if FLAGS.labels_file:
        #lines = open_fn(FLAGS.labels_file, encoding='utf-8').read().splitlines()
        lines = open_fn(FLAGS.labels_file, encoding='gbk').read().splitlines()
        id_2_label = {}
        for line in lines:
            sep = line.split(":")
            if len(sep) == 2:
                id_2_label[int(sep[0])] = sep[1]

        #print id_2_label
        output_count = 5
        print("*******************************************************************")
        for i in range(len(index)-1, len(index)-1-output_count, -1):
            print("%s %s %s" % (index[i], id_2_label[index[i]], scores[index[i]]))

