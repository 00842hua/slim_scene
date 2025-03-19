# -*- coding: utf-8 -*-

import sys
import os
import numpy as np

try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except NameError:
    from imp import reload
    reload(sys)

if len(sys.argv) < 4:
    print("Need param(GTfile PredictFile label_file)")
    exit()
    
gt_file = sys.argv[1]
predict_file = sys.argv[2]
label_file = sys.argv[3]
basename = os.path.basename(predict_file)
dirname = os.path.dirname(predict_file)
badcase_file = os.path.join(dirname, "badcases_" + basename)
badcase_lines = []

gt_map = {}
try:
    gt_lines = open(gt_file).read().splitlines()
except UnicodeDecodeError:
    gt_lines = open(gt_file, encoding='utf-8').read().splitlines()
for line in gt_lines:
    sep = line.split()
    gt_map[sep[1]] = sep[2:]
    
print("len(gt_map): %d " % len(gt_map))

label_map = {}
try:
    gt_lines = open(label_file).read().splitlines()
except UnicodeDecodeError:
    gt_lines = open(label_file, encoding='utf-8').read().splitlines()
for line in gt_lines:
    sep = line.split(":")
    label_map[int(sep[0])] = sep[1]
print("len(label_map): %d " % len(label_map))


total = 0
total_R = 0
TP_byscene = {}
FN_byscene = {}
FP_byscene = {}


def add_statistics(statistic_map, predict_label):
    if predict_label in statistic_map:
        statistic_map[predict_label] += 1
    else:
        statistic_map[predict_label] = 1

try:
    predict_lines = open(predict_file).read().splitlines()
except UnicodeDecodeError:
    predict_lines = open(predict_file, encoding='utf-8').read().splitlines()
for line in predict_lines:
    sep = line.split()
    if len(sep) < 3:
        print("Error Predict line: %s" % line)
        continue
    img_path = sep[0]
    predict = np.array(sep[1:], dtype=np.float32)
    result = np.argmax(predict)
    if img_path not in gt_map:
        print("img_path not in GTFile! %s" % img_path)
        continue
    total += 1
    gts = gt_map[img_path]
    gts_array = [ int(item) for item in gt_map[img_path]]
    if result in gts_array:
        total_R += 1
        add_statistics(TP_byscene, result)
    else:
        predict_result="\t".join(sep[1:])
        badcase_lines.append(img_path+"\t"+"#".join(gts)+"\t"+predict_result+"\n")
        add_statistics(FP_byscene, result)
        for one_gt in gts_array:
            add_statistics(FN_byscene, one_gt)
            
            
            
print("-- total:   %d" % total)            
print("-- total_R: %d" % total_R)            
print("-- len(TP_byscene): %d" % len(TP_byscene))
print("-- len(FN_byscene): %d" % len(FN_byscene))
print("-- len(FP_byscene): %d" % len(FP_byscene))
print("--------------------------------------------------------------------\n")
scene_precision_total = 0.0
scene_precision_count = 0

for scene in FN_byscene:
    if scene not in TP_byscene:
        TP_byscene[scene] = 0

for scene in TP_byscene:
    if scene not in FN_byscene:
        FN_byscene[scene] = 0
    if scene not in FP_byscene:
        FP_byscene[scene] = 0
    gap = "\t"
    if len(label_map[scene]) < 15:
        gap += "\t"
    if len(label_map[scene]) < 10:
        gap += "\t"
    if len(label_map[scene]) < 4:
        gap += "\t"
        
    if (TP_byscene[scene] + FN_byscene[scene] == 0):
        scene_recall = 0
    else:
        scene_recall = 100.0 * TP_byscene[scene] / (TP_byscene[scene] + FN_byscene[scene])
    if (TP_byscene[scene] + FP_byscene[scene] == 0):
        scene_precision = -1
    else:
        scene_precision = 100.0 * TP_byscene[scene] / (TP_byscene[scene] + FP_byscene[scene])
    if scene_precision != -1:
        scene_precision_total += scene_precision
        scene_precision_count += 1

    print("%s%sr:%.2f%% \tp:%.2f%%" % (label_map[scene], gap, scene_recall, scene_precision))
print("--------------------------------------------------------------------\n")
#for scene in FN_byscene:
#    if scene not in TP_byscene:
#        print("FN_byscene %s %d" % (scene, FN_byscene[scene]))
#for scene in FP_byscene:
#    if scene not in TP_byscene:
#        print("FP_byscene %s %d" % (scene, FP_byscene[scene]))
#print("--------------------------------------------------------------------\n")
print("--Total Recall:          %.2f%%" % (100.0 * int(total_R) / int(total)))
print("--SceneAVG Precision:    %.2f%%" % (scene_precision_total / scene_precision_count))


if sys.version_info.major == 2:
    open(badcase_file, "w").writelines(badcase_lines)
else:
    open(badcase_file, "w", encoding='utf-8').writelines(badcase_lines)
