import tensorflow as tf
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
 
sess = tf.Session()
saver = tf.train.import_meta_graph(sys.argv[1]+".meta")
 
#saver.restore(sess, sys.argv[1])
 
graph = tf.get_default_graph()
 
#print(graph.get_operations())
 

opOutLines = []
for op in sess.graph.get_operations():
    opOutLines.append(op.name+"\n")
open('/tmp/opNameCKPT.txt','w').writelines(opOutLines)
