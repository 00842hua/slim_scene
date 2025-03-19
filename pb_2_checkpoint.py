import tensorflow as tf
from nets import nets_factory

pb_file = "D:/TrainedModels/ASPRETRAIN/mobilenet_v2_qc/frozen_mobilenet_v2_1.0_224_wot.pb"
out_ckpt = "D:/TrainedModels/ASPRETRAIN/mobilenet_v2_qc/mobilenet_v2_qc_1.0_224_wot"
#pb_file = "/data_old/ffwang/banknote/20180718_142/CKPT_InResV2_142/Banknote_IncResV2_142_54599.pb"
#out_ckpt = "/data_old/ffwang/banknote/20180718_142/CKPT_InResV2_142/restore-54599"

graph_def = tf.GraphDef()
with open(pb_file, 'rb') as f:
    graph_def.ParseFromString(f.read())

tf.import_graph_def(graph_def, name='load')
graph = tf.get_default_graph()
pb_names = []
for op in graph.get_operations():
    if 'read' in op.name or 'FusedBatchNorm' in op.name or 'Relu' in op.name or 'Predictions' in op.name or 'Conv2D' in op.name or 'BiasAdd' in op.name \
    or 'AvgPool' in op.name or 'Identity' in op.name or 'add' in op.name or 'SpatialSqueeze' in op.name or 'input' in op.name or '/MobilenetV1/MobilenetV1/' in op.name:
        continue
    else:
        print(op.name)
        pb_names.append(op.name)
print(len(pb_names))

input_size = 224
num_classes = 1001

input_node = tf.placeholder(tf.float32, shape=(None, input_size, input_size, 3))
inres = nets_factory.get_network_fn('mobilenet_v2_qc', num_classes)
logits, endpoints = inres(input_node)


train_vars = tf.contrib.framework.get_variables_to_restore()
#train_vars = tf.contrib.framework.get_trainable_variables()
train_vars = [var for var in train_vars if var.name.find('AuxLogits')<0]
print('len(train_vars): {}'.format(len(train_vars)))
graph = tf.get_default_graph()
assign_ops = []
for idx, var in enumerate(train_vars):
    print(var.name)
    assign_ops.append(tf.assign(var, graph.get_tensor_by_name("%s:0" % pb_names[idx]))) #var.name)))

sess = tf.Session()
_ = sess.run(assign_ops)

saver = tf.train.Saver(train_vars)
_ = saver.save(sess, out_ckpt)