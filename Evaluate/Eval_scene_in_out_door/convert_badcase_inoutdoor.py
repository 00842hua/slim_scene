import sys
import numpy as np

# python /home/nas01/grp_IMRECOG/wangfei11/Code/slim_scene/Evaluate/Eval_scene_1k/convert_badcase_1k.py badcases_result_score_136195.txt

if len(sys.argv) < 2:
    print("Need badcase file!")
    exit(1)

infile = sys.argv[1]
outfile = sys.argv[1] + ".out"
labelfile = "/home/work/wangfei11/data/AI_scene_1k/labels_1609.txt"

labellines = open(labelfile).read().splitlines()
labelmap = {}
for line in labellines:
    sep = line.split(":")
    if len(sep) == 2:
        labelmap[int(sep[0])] = sep[1]

print("infile: %s" % infile)
print("outfile: %s" % outfile)
print("labelfile: %s" % labelfile)
print("len(labelmap): %d" % len(labelmap))

outlines = []
lines = open(infile).read().splitlines()
for line in lines:
    sep = line.split()
    scores = np.array(sep[2:])
    scores = [float(item) for item in scores]
    #print("len(sep): %d" % len(sep))
    #print("len(scores): %d" % len(scores))
    # proc multi label GT like "161#370"
    sep_gt = sep[1].split("#")
    gt_label = labelmap[int(sep_gt[0])]
    predict = labelmap[np.argmax(scores)]
    score = np.max(scores)
    outline = sep[0] + "\t" + gt_label + "\t" + predict + "\t" + str(score) + "\n"
    outlines.append(outline)
    #break
    
open(outfile, 'w').writelines(outlines)

    
