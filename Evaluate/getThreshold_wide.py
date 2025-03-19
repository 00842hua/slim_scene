# -*- coding: utf-8 -*-

import sys
import numpy as np
import time
import os

tp_count = {}
fp_count = {}
fn_count = {}
p_count = {}
r_count = {}
gtdict = {}
labeldict = {}

if len(sys.argv) < 5:
    print("Usage: %s result_file_path gt_file_path label_num label_path" % sys.argv[0])
    exit()
result_file_path = sys.argv[1]
gt_file_path = sys.argv[2]
clsnum = int(sys.argv[3])
label_path = sys.argv[4]
filenamesep = os.path.splitext(result_file_path)
thres_result_file = filenamesep[0]+"_thres"+ filenamesep[1]

print("result_file_path: %s" % result_file_path)
print("thres_result_file: %s" % thres_result_file)


for line in open(label_path, 'r'):
    line = line.strip().split('\t')
    labeldict[int(line[1])] = line[0]

th_me = np.zeros(clsnum)
# read gtlist
gtdict = {}
for line in open(gt_file_path, 'r'):
    line = line.strip().split('\t')
    gtdict[line[1]] = line[2:]

f_metric = np.zeros((clsnum, 96))

start = time.time()
lines = open(result_file_path,'r').read().splitlines()
# read label
for i in range(clsnum):
    for j in range(5, 96, 1):
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
            f_metric[i,j] = F_helf

for i in range(clsnum):
    index = np.argmax(f_metric[i,:])
    th_me[i] = round(index/100.0, 2)
    if th_me[i] == 0:
        th_me[i] = 0.5

outlines = []
for i in range(clsnum):
    oneline = "%s\t%s\n"%(labeldict[i],str(th_me[i]))
    outlines.append(oneline)
    #print '\'{}\':{},'.format(labeldict[i],str(th_me[i])),
    #if (i+1) % 6==0:
    #  print ''
open(thres_result_file, "w").writelines(outlines)

print 'Computation time: {0}'.format(time.time() - start)
