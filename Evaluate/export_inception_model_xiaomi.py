#-*- coding:utf-8 -*-

import tensorflow as tf
import sys

sys.path.append("../")
sys.path.append("/home/nas01/grp_IMRECOG/wangfei11/Code/slim_scene/")
sys.path.append("G:/CameraScene/slim_scene/")

from nets import nets_factory
#from datasets import dataset_factory

if tf.__version__.startswith('1.'):
    from tensorflow.python.tools import freeze_graph
else:
    import freeze_graph

slim = tf.contrib.slim

'''
python export_inception_model.py --check_point=/home/work/wangfei11/TF/ckpt_cls34_20190314_mobilev1_gaotong/model.ckpt-313 \
--graph_path=/home/work/wangfei11/TF/ckpt_cls34_20190314_mobilev1_gaotong/model.ckpt-313.pb --labels_number=34 --net_type=mobilenet_v1_qc

/c/Anaconda365/python G:/CameraScene/slim_scene/Evaluate/export_inception_model_xiaomi.py \
--check_point=model.ckpt-34191 \
--graph_path=model.ckpt-34191.pb \
--labels_number=5 \
--net_type=mobilenet_v1_qc
'''

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('check_point', None, 'checkpoint file')
tf.app.flags.DEFINE_string('graph_path', None, 'graph path')
tf.app.flags.DEFINE_string('net_type', None, 'model name in nets factory')
tf.app.flags.DEFINE_integer('image_size', None, 'input image size')
tf.app.flags.DEFINE_integer('channel', 3, 'input image channel')
tf.app.flags.DEFINE_integer('labels_number', 323, 'input class_num ')
tf.app.flags.DEFINE_bool('quantize', False, 'quantize in PB.')
tf.app.flags.DEFINE_boolean('inres_use_aux', True, 'inception_resnet_v2 use inres aux')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_string('output_node_names', None, 'output_node_names')

'''
python export_for_PNAS.py \
    --check_point=./pre_model/model.ckpt \
    --net_type=pnasnet_large \
    --labels_number=1001 \
    --image_size=331 \
    --graph_path=./pre_model/default.pb \
    --output_node=Predictions
'''


if __name__ == "__main__":
    if not FLAGS.check_point:
        raise ValueError('need set check_point')

    if not FLAGS.graph_path:
        raise ValueError('need set graph_path')

    if not FLAGS.net_type:
        raise ValueError('need set model_name')

    sess = tf.Session()

    # 多标签和单标签共用网络结构，只是在prediction的时候使用softmax和sigmoid的区别
    net_type = FLAGS.net_type
    if net_type == "inception_resnet_v2_multilabel":
        net_type = "inception_resnet_v2"
    if net_type == "nasnet_large_multilabel":
        net_type = "nasnet_large"
    if net_type == "nasnet_mobile_multilabel":
        net_type = "nasnet_mobile"


    # mobilenet v1 只输出logits，为了适配MTK的apu，他们不支持sigmoid操作。网络结构还是一样的
    if net_type == "mobilenet_v1_qc_logits":
        net_type = "mobilenet_v1_qc"

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
            output_node_names = "InceptionResnetV2/Logits/Predictions_Mix_Aux,InceptionResnetV2/Logits/Predictions"
    elif FLAGS.net_type == 'inception_resnet_v2_multilabel':
        new_logit = end_points['Logits']
        inres_output = tf.nn.sigmoid(new_logit, name='InceptionResnetV2/Logits/Predictions_multilabel')
        output_node_names = "InceptionResnetV2/Logits/Predictions_multilabel,InceptionResnetV2/Logits/Predictions"
        if FLAGS.inres_use_aux:
            print("inres use aux")
            new_logit = (end_points['AuxLogits'] + end_points['Logits']) / 2.0
            inres_output_mix = tf.nn.sigmoid(new_logit, name='InceptionResnetV2/Logits/Predictions_Mix_Aux')
            output_node_names = "InceptionResnetV2/Logits/Predictions_Mix_Aux,InceptionResnetV2/Logits/Predictions_multilabel,InceptionResnetV2/Logits/Predictions"
    elif FLAGS.net_type == 'inception_v3':
        output_node_names = "InceptionV3/Predictions/Softmax"
    elif FLAGS.net_type == 'resnet_v1_50':
        output_node_names = "resnet_v1_50/predictions/Softmax"
    elif FLAGS.net_type == 'resnet_v2_50':
        output_node_names = "resnet_v2_50/predictions/Softmax"
    elif FLAGS.net_type == 'resnet_v2_101':
        output_node_names = "resnet_v2_101/predictions/Softmax"
    elif FLAGS.net_type == 'resnet_v2_152':
        output_node_names = "resnet_v2_152/predictions/Softmax"
    elif FLAGS.net_type == 'inception_v4':
        output_node_names = "InceptionV4/Logits/Predictions"
    elif FLAGS.net_type == 'nasnet_large' or FLAGS.net_type == 'nasnet_mobile':
        output_node_names = "final_layer/predictions"
    elif FLAGS.net_type == 'nasnet_large_multilabel' or FLAGS.net_type == 'nasnet_mobile_multilabel':
        output_node_names = "final_layer/predictions_sigmoid"
    elif FLAGS.net_type == 'mobilenet_v2' or FLAGS.net_type == 'mobilenet_v2_qc':
        output_node_names = "MobilenetV2/Predictions/Reshape_1,MobilenetV2/Predictions_sigmoid,MobilenetV2/Predictions_softmax"
    elif FLAGS.net_type == 'mobilenet_v2_multilabel':
        output_node_names = "MobilenetV2/Predictions"
    elif FLAGS.net_type == 'mobilenet_v2_140_multilabel':
        output_node_names = "MobilenetV2/Predictions"
    elif FLAGS.net_type == 'mobilenet_v1_qc' or FLAGS.net_type == 'mobilenet_v1_qc_se':
        output_node_names = "MobilenetV1/Predictions,MobilenetV1/Predictions_sigmoid,MobilenetV1/Predictions_softmax"
    elif FLAGS.net_type == 'mobilenet_v1_qc_logits':
        output_node_names = "MobilenetV1/Logits/SpatialSqueeze"
    elif FLAGS.net_type == 'mobilenet_v1_multilabel':
        output_node_names = "MobilenetV1/Predictions"
    elif FLAGS.net_type == 'mobilenet_v3_large':
        output_node_names = "MobilenetV3_large/Logits/Predictions_sigmoid"
    elif FLAGS.net_type == 'mobilenet_v3_large_new':
        output_node_names = "MobilenetV3Large/Predictions_softmax,MobilenetV3Large/Predictions_sigmoid"
    elif FLAGS.net_type == 'pelee_net_multilabel':
        output_node_names = "pelee_net/Logits/Predictions"
    elif FLAGS.net_type == 'shufflenet_multilabel':
        output_node_names = "ShuffleNet/Predictions"
    elif FLAGS.net_type == 'shufflenet_v2_multilabel':
        output_node_names = "Shufflenet_v2/Predictions"
    elif FLAGS.net_type == 'shufflenet_v2d15_multilabel':
        output_node_names = "Shufflenet_v2/Predictions"
    elif FLAGS.net_type == 'densenet_121_multilabel':
        output_node_names = "DenseNet_121/predictions"
    elif FLAGS.net_type == 'efficientnet_b0':
        output_node_names = "efficientnet_b0/Predictions_softmax,efficientnet_b0/Predictions_sigmoid"
    elif FLAGS.net_type == 'efficientnet_b0_wf':
        output_node_names = "efficientnet_b0/Logits/Predictions_sigmoid"
    else:
        if FLAGS.output_node_names:
            output_node_names = FLAGS.output_node_names
        else:
            raise ValueError('net type not support')

    #output_node_names='final_layer/predictions'

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
    else:
      variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    
    opOutLines = []
    for op in sess.graph.get_operations():
        opOutLines.append(op.name+"\n")
    open('/tmp/opName.txt','w').writelines(opOutLines)

    # for op in sess.graph.get_operations():
    #     if not op.name.startswith('save'):
    #         print(op.name + "\n")

    # 导出图模型，不带参数
    tf.train.write_graph(sess.graph_def, "/tmp/", "temp.pb", as_text=False)

    # 指定具体的checkpoint文件
    input_checkpoint_path = FLAGS.check_point
    saver.restore(sess, input_checkpoint_path)  # 加载checkpoint的参数到图模型

    input_graph_path = "/tmp/temp.pb"
    input_saver_def_path = ""
    input_binary = True

    # 指定输出节点的name
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_path = FLAGS.graph_path
    clear_devices = False

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, input_checkpoint_path,
                              output_node_names, restore_op_name,
                              filename_tensor_name, output_graph_path,
                              clear_devices, "")
