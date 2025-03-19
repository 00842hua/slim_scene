# -*- coding: utf-8 -*-

import sys
import numpy as np
import time
import os
import multiprocessing

# ONLY work on Linux/Unix 
# python /home/work/wangfei11/slim_scene/Evaluate/Eval_video_label/getThreshold_video_tag_mul_thread.py result_score_159468.txt /home/work/wangfei11/data/video_labeling/Test_IMGS_SmallCate/imListTest_gt_149.txt 149 /home/work/wangfei11/data/video_labeling/labels_149.txt 


tp_count = {}
fp_count = {}
fn_count = {}
p_count = {}
r_count = {}
gtdict = {}
labeldict = {}

if len(sys.argv) < 5:
    print("Usage: %s result_file_path gt_file_path label_num label_path THREAD_NUM" % sys.argv[0])
    exit()
result_file_path = sys.argv[1]
gt_file_path = sys.argv[2]
clsnum = int(sys.argv[3])
label_path = sys.argv[4]
THREAD_NUM=10
DEFAULT_THRESHOLD=1.0
if len(sys.argv) >= 6:
    THREAD_NUM = int(sys.argv[5])
    
filenamesep = os.path.splitext(result_file_path)
thres_result_file = filenamesep[0]+"_thres"+ filenamesep[1]

print("result_file_path: %s" % result_file_path)
print("thres_result_file: %s" % thres_result_file)
print("THREAD_NUM: %s" % THREAD_NUM)


for line in open(label_path, 'r'):
    line = line.strip().split('\t')
    labeldict[int(line[1])] = line[0]

th_me = np.zeros(clsnum)
# read gtlist
gtdict = {}
gt_labels_set = set()
for line in open(gt_file_path, 'r'):
    line = line.strip().split('\t')
    gtdict[line[1]] = line[2:]
    for item in line[2:]:
        gt_labels_set.add(item)
        
print("len(gt_labels_set) : %d " % (len(gt_labels_set)))
print(gt_labels_set)

#f_metric = np.zeros((clsnum, 96))


start = time.time()
lines = open(result_file_path,'r').read().splitlines()
MAX_THRES = 95  # MAXIMUM OF THRESHOLD

def _calc_thres_thread(thread_id, thread_num, f_metric_share):
    for i in range(clsnum):
        if i % thread_num != thread_id:
            continue
        print("-------- thread[%d : %d]  %d / %d" % (thread_id, thread_num, i, clsnum))
        if str(i) not in gt_labels_set:
            continue
        for j in range(5, MAX_THRES, 5):
            th_me[i] = j/100.0
            tp_count[i] = 0
            fp_count[i] = 0
            fn_count[i] = 0
            p_count[i] = 0
            r_count[i] = 0

            # read score
            for line in lines:
                line = line.strip().split('\t')
                #name = line[0].split('/')[-1]  # 现在使用全路径
                name = line[0]
                score = float(line[i+1]) - th_me[i]
                gts = gtdict[name]
                if str(i) in gts:
                    if score > 0:
                        tp_count[i] += 1
                    else:
                        fn_count[i] += 1
                else:
                    if score > 0:
                        fp_count[i] += 1

            if tp_count[i] != 0:
                p_count[i] = float(tp_count[i]) / (tp_count[i] + fp_count[i])
                r_count[i] = float(tp_count[i]) / (tp_count[i] + fn_count[i])
                F_helf = 1.25 * p_count[i] * r_count[i] / (0.25 * p_count[i] + r_count[i])
                #F_helf = 1.25 * p_count[i] * r_count[i] / (0.25 * (p_count[i] + r_count[i]))
                #f_metric_share[i,j] = F_helf
                f_metric_share[i*MAX_THRES+j] = F_helf


if __name__ == '__main__':

    multiprocessing.freeze_support()
    
    f_metric_share=multiprocessing.Array("f", clsnum*MAX_THRES)
    thread_list = []
    for i in range(THREAD_NUM):
        sthread = multiprocessing.Process(target=_calc_thres_thread, args=(i, THREAD_NUM, f_metric_share))
        #sthread.setDaemon(True)
        sthread.start()
        thread_list.append(sthread)
    for i in range(THREAD_NUM):
        thread_list[i].join()

    f_metric_array = [ i for i in f_metric_share]
    #print(f_metric_array[15*MAX_THRES:16*MAX_THRES])
    f_metric = np.array(f_metric_array)
    #print(f_metric[15*MAX_THRES:16*MAX_THRES])
    f_metric = f_metric.reshape(clsnum, -1)
    #print("f_metric.shape:")
    #print(f_metric.shape)
    #print(f_metric[15, :])

    for i in range(clsnum):
        index = np.argmax(f_metric[i, :])
        th_me[i] = round(index/100.0, 2)
        if th_me[i] == 0:
            th_me[i] = DEFAULT_THRESHOLD

    outlines = []
    for i in range(clsnum):
        oneline = "%s\t%s\n"%(labeldict[i],str(th_me[i]))
        outlines.append(oneline)
        #print '\'{}\':{},'.format(labeldict[i],str(th_me[i])),
        #if (i+1) % 6==0:
        #  print ''
    open(thres_result_file, "w").writelines(outlines)

    print('Computation time: %d' %(time.time() - start))
