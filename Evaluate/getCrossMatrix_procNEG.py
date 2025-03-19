# -*- coding: utf-8 -*-


# 用于多标签标注后的测试集的精度验证，生成result_score的gtList应该是imListTest_gt_43.txt，也用这个文件生成阈值
# imListTest_gt_43_multi_label_md5.txt 是记录每个md5值对应文件的gt列表，这个是多标签标注后的，用于计算实际精度。
# 用md5记录的原因是同一个图片会有多个目录里有，每次生成 imListTest_gt_43.txt 的时候会使用到不同图片，但是md5是不变的

'''
python -u /home/work/wangfei11/slim_scene/Evaluate/getCrossMatrix_procNEG.py \
result_score_55577.txt \
/home/work/wangfei11/data/AI_scene_back/Test_IMGS/imListTest_gt_43_multi_label_md5.txt \
43 \
/home/work/wangfei11/data/AI_scene_back/label_43.txt \
/home/work/wangfei11/slim_scene/Evaluate/Eval_label_map.txt
'''

import sys
import numpy as np
import os
np.set_printoptions(threshold=np.inf)

if len(sys.argv) < 5:
    print("Usage: %s result_file_path gt_file_path label_num label_path [label_map_file]" % sys.argv[0])
    exit()

tp_alldim_count = {}
fp_alldim_count = {}
tp_count = {}
fp_count = {}
fn_count = {}
p_count = {}
p_alldim_count = {}
r_count = {}
img_gtdict = {}
labeldict = {}
labeldict_name2idstr = {}
score_dic = {}
gtdict = {}
name_list = []
tp_dic = {}
r_dic = {}
p_dic = {}
p_list = []
score_path = sys.argv[1]
gt_path = sys.argv[2]
clsnum = int(sys.argv[3])
label_path = sys.argv[4]
crossmatrix = np.zeros( (clsnum+1, clsnum+1), dtype=np.int)

# mapping labels, eg: {{'beach': ['waterside', 'scenery']}, {'food': ['curry']}}
label_map_dict = {}
label_map_dict_id = {}


# gt_path 文件中对反例使用特别的编号9999，读到后替换为 clsnum ，这样在增加类型后，如果没有增加测试用例，不用调整 gt_path 文件
NEG_IDX = 9999

thres_file_path = os.path.splitext(score_path)[0]+"_thres"+ os.path.splitext(score_path)[1]
result_name = os.path.splitext(score_path)[0]+"_matrix"+ os.path.splitext(score_path)[1]
result_badcase_file = os.path.splitext(result_name)[0]+"_badcase"+ os.path.splitext(result_name)[1]
result_badcase_file_alldim = os.path.splitext(result_name)[0]+"_alldim_badcase"+ os.path.splitext(result_name)[1]

thresholds = {}
thres_lines = open(thres_file_path).read().splitlines()
for line in thres_lines:
    sep = line.split()
    if len(sep) == 2:
        thresholds[sep[0]] = float(sep[1])


for line in open(label_path,'r'):
    line = line.strip().split('\t')
    #print("%s %s" % (line, len(line)))
    labeldict[int(line[1])] = line[0]
    labeldict_name2idstr[line[0]] = line[1]
    

if len(sys.argv) >= 6:
    result_name = os.path.splitext(score_path)[0]+"_matrix_labelmapping"+ os.path.splitext(score_path)[1]
    result_badcase_file = os.path.splitext(result_name)[0]+"_badcase"+ os.path.splitext(result_name)[1]
    result_badcase_file_alldim = os.path.splitext(result_name)[0]+"_alldim_badcase"+ os.path.splitext(result_name)[1]
    label_map_file = sys.argv[5]
    for line in open(label_map_file, 'r'):
        sep = line.strip().split()
        if len(sep) != 2:
            print("%s error line: %s" % (label_map_file, line))
            continue
        mapping_gts = sep[1].split('|')
        label_map_dict[sep[0]] = mapping_gts
    for k in label_map_dict:
        v = [labeldict_name2idstr[item] for item in label_map_dict[k] if item in labeldict_name2idstr]
        label_map_dict_id[labeldict_name2idstr[k]] = v
    



print("score_path: %s" % score_path)
print("thres_file_path: %s" % thres_file_path)
print("result_name: %s" % result_name)
print(thresholds)
print("------------------ label_map_dict ------------------")
for k in label_map_dict:
    print("%s:\t%s" % (k, label_map_dict[k]))
print("------------------ label_map_dict_id ------------------")
for k in label_map_dict_id:
    print("%s:\t%s" % (k, label_map_dict_id[k]))

'''           
thresholds = {'document':0.4, 'flower':0.1, 'food':0.35, 'ppt':0.1, 'sky':0.71, 'sunset':0.1, 
'cat':0.55, 'dog':0.39, 'plant':0.79, 'nightscene':0.35, 'snow':0.1, 'waterside':0.28, 
'autumn':0.14, 'candlelight':0.15, 'car':0.1, 'lawn':0.19, 'redleaf':0.27, 'succulent':0.29, 
'building':0.39, 'cityscape':0.62, 'cloud':0.26, 'cloudy':0.14, 'accessory':0.13, 'buddha':0.15, 
'cow':0.11, 'curry':0.13, 'motor':0.1, 'temple':0.13, 'beach':0.33, 'diving':0.2, 
'tower':0.26, 'moon':0.14, 'sideface':0.78, 'statue':0.13}
'''


th_me = [thresholds[labeldict[i]] for i in range(clsnum)]

for i in range(clsnum):
    tp_count[i] = 0
    fp_count[i] = 0
    tp_alldim_count[i] = 0
    fp_alldim_count[i] = 0
    fn_count[i] = 0
    p_count[i] = 0
    p_alldim_count[i] = 0
    r_count[i] = 0
    gtdict[i] = []    # key是场景id(0 ~ classnum-1), value是有这个场景标签的测试图片列表
    score_dic[i] = [] # key是场景id(0 ~ classnum-1), value是1.5w测试图预测值在这个类的置信度列表
    tp_dic[i] = 0
    r_dic[i] = []
    p_dic[i] = []
    
    
    
# 对 gts 进行扩展，比如 beach 的case如果识别为waterside、scenery是正确的
# 假设某个case只标注了beach，它的gts是 ['beach'], 这里会扩展为 ['beach', 'watersize', 'scenery']
# 上面是示例，真正的gts应该是保存的labelid
def extendGTS(gts):
    gts_extended = gts[:]
    for gt in gts:
        if gt in label_map_dict_id:
            gts_extended.extend(label_map_dict_id[gt])
    return gts_extended


for line in open(gt_path,'r'):
    line = line.strip().split('\t')
    gts = line[2:]
    # 反例可以有其他标签，说明它识别为反例ok，识别为其他标注的标签也ok。20191017 wangfei
    #if len(gts) == 1 and int(gts[0]) == int(NEG_IDX):
    #    gts = [str(clsnum)]
    for i in range(len(gts)):
        if int(gts[i]) == int(NEG_IDX):
            gts[i] = str(clsnum)
            
    img_gtdict[line[1]] = gts
    #for gt in extendGTS(gts):
    for gt in gts:
        if gt == str(clsnum):
            continue
        gtdict[int(gt)].append(line[1])

#print("len(gtdict[39]): %s " % len(gtdict[39]))
    

badcases_lines = []
badcases_lines_alldim = []
# read score
for line in open(score_path,'r'):
    line = line.strip().split('\t')
    #name = line[0].split('/')[-1]  # 测试图片名称
    name = line[0]  # 测试图片名称
	md5 = name.split("#")[-1].split(".")[0]
    score = np.max(np.squeeze([float(x) for x in line[1:]]) - np.squeeze(th_me))
    pred = np.argmax(np.squeeze([float(x) for x in line[1:]]) - np.squeeze(th_me))

    name_list.append(name)
    score_ = np.squeeze([float(x) for x in line[1:]])
    for i, x in enumerate(score_):
        score_dic[i].append(x)

    gts = img_gtdict[md5]
    gts_labelname = [labeldict[int(item)] for item in gts]
        
    ## 这部分是对结果每一维进行考察，如果这维的置信度高于阈值，则如果gts包含当前标签，就记tp，否则记fp，最后只计算准确率。wangfei 20191018
    scores = np.squeeze([float(x) for x in line[1:]]) - np.squeeze(th_me)
    for idx in range(len(scores)):
        if scores[idx] <= 0:
            continue
        if str(idx) in extendGTS(gts):
            tp_alldim_count[idx] += 1
        else:
            fp_alldim_count[idx] += 1
            badcases_lines_alldim.append(name + "\t" + "#".join(gts_labelname) + "\t" + \
                                  labeldict[idx] + "\t" + str(scores[idx]) + "\n")
            
    ## 这部分只考察top1的情况
    ## 正例的情况： 若top1的pred在gts中，记tp(pred)；若pred不在gts中，记fp(pred)并且gts所有的标签都记fn
    ## 反例的情况： 若top1有预测pred，记fp(pred)
    if score > 0:
        if str(pred) in extendGTS(gts):
            tp_count[pred] += 1
            crossmatrix[pred][pred] += 1
        else:
            fp_count[pred] += 1
            if len(gts)== 1 and gts[0] == str(clsnum):
                crossmatrix[clsnum][pred] += 1
            else:
                for label_count in range(len(gts)):
                    #print name
                    if int(gts[label_count])<clsnum:
                        fn_count[int(gts[label_count])] += 1
                        crossmatrix[int(gts[label_count])][pred] += 1
            badcases_lines.append(name + "\t" + "#".join(gts_labelname) + "\t" + \
                                  labeldict[pred] + "\t" + str(score) + "\n")
    else:
        if str(clsnum) not in gts:
            for label_count in range(len(gts)):
                #print name
                assert int(gts[label_count])<clsnum
                fn_count[int(gts[label_count])] += 1
                crossmatrix[int(gts[label_count])][clsnum] += 1
            badcases_lines.append(name + "\t" + "#".join(gts_labelname) + "\t" + "NEG" + "\t" + str(score) + "\n")
        else:
            crossmatrix[clsnum][clsnum] += 1

open(result_badcase_file, 'w').writelines(badcases_lines)
open(result_badcase_file_alldim, 'w').writelines(badcases_lines_alldim)



def check_if_img_name_in_gtdict(img_name, labelid):
    if img_name in gtdict[labelid]:
        return True
    if str(labelid) in label_map_dict_id:
        for item in label_map_dict_id[str(labelid)]:
            if img_name in gtdict[int(item)]:
                return True
                
    return False
    

    #for label_count in range(len(gts)):
    #    crossmatrix[int(gts[label_count])][pred] += 1
for i in score_dic.keys():
    score_list = score_dic[i]
    score_index = np.argsort(-np.squeeze(score_list))
    for j in range(len(score_index)):
        img_name = name_list[score_index[j]]
        #if img_name in gtdict[i]:
        if check_if_img_name_in_gtdict(img_name, i):
            tp_dic[i] += 1
            p_dic[i].append(tp_dic[i]/float(j+1))
            r_dic[i].append(tp_dic[i]/float(len(gtdict[i])))
        if tp_dic[i] == len(gtdict[i]):
            break
    for k in range(len(p_dic[i])-1):
        if p_dic[i][k] < p_dic[i][k+1]:
           p_dic[i][k] = p_dic[i][k+1]
    p_list.append(np.mean(np.squeeze(p_dic[i])))
p_list_ori = [x for x in p_list if str(x) != 'nan']   # 如果新增类型，还没有对应测试集，可能存在为 nan 的情况
p_list = [x for x in p_list]
mAP = np.mean(np.squeeze(p_list_ori))

for kk in range(clsnum):
    if tp_count[kk] != 0:
        p_count[kk] = float(tp_count[kk]) / (tp_count[kk] + fp_count[kk])
        r_count[kk] = float(tp_count[kk]) / (tp_count[kk] + fn_count[kk])
        p_alldim_count[kk] = float(tp_alldim_count[kk]) / (tp_alldim_count[kk] + fp_alldim_count[kk])
print tp_count
print fp_count
print fn_count
print p_count
print r_count

average_p = float(np.sum(tp_count.values())) / (np.sum(tp_count.values()) + np.sum(fp_count.values()))
average_r = float(np.sum(tp_count.values())) / (np.sum(tp_count.values()) + np.sum(fn_count.values()))
average_p_alldim = float(np.sum(tp_alldim_count.values())) / (np.sum(tp_alldim_count.values()) + np.sum(fp_alldim_count.values()))
print average_p
print average_r
print crossmatrix
writefile = open(result_name, 'w')
for i in range(clsnum):
    line = labeldict[i][:10].rjust(11) + ' R = ' + '{:03.2f}'.format(r_count[i]*100) + ' P = ' + '{:03.2f}'.format(p_count[i]*100) \
            + ' P_AllDim = ' + '{:3.2f}'.format(p_alldim_count[i]*100)  + ' ERR_AllDim = ' + '{:3.2f}'.format((1-p_alldim_count[i])*100) \
            + ' fp_alldim_count = ' + '{:2d}'.format(fp_alldim_count[i]) + ' tp+fp = ' + '{:3d}'.format(tp_alldim_count[i] + fp_alldim_count[i]) \
            + ' ap = ' + '{:3.2f}'.format(p_list[i]*100) + '\n'
    writefile.writelines(line)
writefile.writelines('average_p = ' + str(average_p) + '\n')
writefile.writelines('average_r = ' + str(average_r) + '\n')
writefile.writelines('mAP = ' + str(mAP) + '\n')
writefile.writelines('average_p_alldim = ' + str(average_p_alldim) + '\n')
line = ' '*13
for i in range(clsnum + 1):
    line += labeldict[i][:3].rjust(4)
writefile.writelines(line+'\n')
for i in range(clsnum + 1):
    line = labeldict[i][:10].rjust(11) + ': '
    for j in crossmatrix[i]:
        line += str(j).rjust(4)
    line += '\n'
    writefile.writelines(line)
writefile.close()
