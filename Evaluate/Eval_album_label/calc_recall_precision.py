# -*- coding: utf-8 -*-

import sys
import codecs

reload(sys) 
sys.setdefaultencoding('utf-8')

if len(sys.argv) < 2:
    print("Need Param result_brief_file [label_map_file]")
    exit()
    
result_file = sys.argv[1]

def add_label_map(k1, k2):
    if k1 not in label_map:
        label_map[k1] = [k2]
    else:
        label_map[k1].append(k2)
    if k2 not in label_map:
        label_map[k2] = [k1]
    else:
        label_map[k2].append(k1)

label_map = {}
if len(sys.argv) >= 3:
    label_map_file = sys.argv[2]
    for line in codecs.open(label_map_file, encoding="UTF-8"):
        sep = line.split("|")
        for i in range(len(sep)):
            for j in range(i+1, len(sep)):
                add_label_map(sep[i], sep[j])
    
gt_set = set()
prediction_set = set()
TP = {}
FP = {}
FN = {}

def collection_inc(collection, key):
    if key in collection:
        collection[key] += 1
    else:
        collection[key] = 1
    

for line in codecs.open(result_file, encoding="UTF-8"):
    sep = line.split("|")
    ground_truth = sep[1]
    prediction = sep[2]
    gt_set.add(ground_truth)
    prediction_set.add(prediction)
    if ground_truth == prediction or (prediction in label_map and ground_truth in label_map[prediction]):
        collection_inc(TP, ground_truth)
    else:
        collection_inc(FN, ground_truth)
        collection_inc(FP, prediction)
        
def calc_gap(k):
    gap="      "
    if len(k)<=6:
        gap="              "
    if len(k)<=3:
        gap="                  "
    return gap
    
    
totalR = 0
totalP = 0
for k in gt_set:
    if k not in TP:
        TP[k] = 0
    if k not in FP:
        FP[k] = 0
    if k not in FN:
        FN[k] = 0
    gap = calc_gap(k)
    if TP[k]+FN[k] == 0:
        recall=0 
    else:
        recall=100.0*TP[k]/(TP[k]+FN[k])
    if (TP[k]+FP[k] == 0):
        precision=0
    else:
        precision=100.0*TP[k]/(TP[k]+FP[k])
    totalR+=recall
    totalP+=precision
    print("%s %s     \t%.2f       \t%.2f" %(k, gap, recall, precision))
print("AVG:     \t\t\t%.2f       \t%.2f" % (float(totalR)/len(gt_set), float(totalP)/len(gt_set)))

if len(sys.argv) >= 4:
    print("--------------------------------------------------------------------------------")
    print("Recall Error:")
    for k in prediction_set:
        if k not in FP or FP[k] == 0:
            continue
        gap = calc_gap(k)
        print ("%s %s     \t%s" % (k, gap, FP[k]))
