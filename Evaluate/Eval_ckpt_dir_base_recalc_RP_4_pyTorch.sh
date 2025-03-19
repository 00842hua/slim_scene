if [ $# -lt 3 ]
then
    echo "Usage: $0 NetType OutputNodeNames CkptDir [proprocess_name img_size LABEL_NUM]"
    exit 1
fi

net_type=$1
output_node_names=$2
dir=$3

preprocess_name="scene"
if [ $# -ge 4 ]
then
    preprocess_name=$4
fi

img_size=224
if [ $# -ge 5 ]
then
    img_size=$5
fi


LABEL_NUM=36
if [ $# -ge 6 ]
then
    LABEL_NUM=$6
fi

TEST_IMG_GT_FILE_DIR=/home/work/wangfei11/data/AI_scene_back/Test_IMGS
if [ $# -ge 7 ]
then
    TEST_IMG_GT_FILE_DIR=$7
fi
TEST_IMG_GT_FILE_PATH=${TEST_IMG_GT_FILE_DIR}/imListTest_gt_${LABEL_NUM}.txt

echo "dir: " ${dir}
echo "preprocess_name: " ${preprocess_name}
echo "img_size: " ${img_size}
echo "LABEL_NUM: " ${LABEL_NUM}
echo "net_type: " ${net_type}
echo "output_node_names: " ${output_node_names}


while true
do
    checkpoint_steps=`ls ${dir}/result_score_*.txt | awk -F"_" '{print $NF}' | cut -d"." -f1 | sort -nr`
    for one_ckpt_step in ${checkpoint_steps[@]}
    do
        score_file=${dir}/result_score_${one_ckpt_step}.txt
        echo "-----------------------------------------------------------------------------" >&2
        echo "`date +"%Y-%m-%d %H:%M:%S"` ${score_file}" >&2
        if [ -e ${score_file} ]
        then
                
            python -u /home/work/wangfei11/slim_scene/Evaluate/getThreshold_wide_mul_thread.py ${score_file} \
                ${TEST_IMG_GT_FILE_PATH}  ${LABEL_NUM} /home/work/wangfei11/data/AI_scene_back/label_${LABEL_NUM}.txt
            
            python -u /home/work/wangfei11/slim_scene/Evaluate/getCrossMatrix.py ${score_file} \
                ${TEST_IMG_GT_FILE_PATH} ${LABEL_NUM} /home/work/wangfei11/data/AI_scene_back/label_${LABEL_NUM}.txt
        else
            echo "`date +"%Y-%m-%d %H:%M:%S"` ${pb_path} Already EXISTS, Will Skip" >&2
        fi
    done
    
    break
done

