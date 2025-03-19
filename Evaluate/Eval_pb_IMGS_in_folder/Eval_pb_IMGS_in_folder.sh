# 用于评测放在文件夹里的图片，比如 /a/b/c 目录下
# 该目录下，每个类型的图片需要在独立子文件夹中，比如 /a/b/c/flower   /a/b/c/document  文件夹名称需要和labels文件对得上


# /home/work/wangfei11/slim_scene/Evaluate/Eval_pb_IMGS_in_folder.sh pb_path output_node_names preprocess_name img_size EVAL_IMGS_DIR LABEL_FILE
:<<!

/home/work/wangfei11/slim_scene/Evaluate/Eval_pb_IMGS_in_folder.sh \
/home/nas01/grp_IMRECOG/TF_RECORD/AI_scene_back/TF_40_20200601/CKPT/CKPT_MV1QC_mixup_bs200_lrconf_1gpu_quant_89_1/model.ckpt-93740.quant.pb \
MobilenetV1/Predictions \
scene \
224 \
/home/work/wangfei11/data/AI_scene_back/Test_IMGS_Special/20200604_doc_test \
/home/work/wangfei11/data/AI_scene_back/label_40.txt \
/home/nas01/grp_IMRECOG/TF_RECORD/AI_scene_back/TF_40_20200601/CKPT/CKPT_MV1QC_mixup_bs200_lrconf_1gpu_quant_89_1/result_score_93740_thres.txt 


/home/work/wangfei11/slim_scene/Evaluate/Eval_pb_IMGS_in_folder.sh \
/home/nas01/grp_IMRECOG/wangfei11/data_66/OnlineModel/scene_zhongkong_10/20200609/model.ckpt-36756.quant.pb \
MobilenetV1/Predictions \
scene \
224 \
/home/work/wangfei11/data/AI_scene_back/Test_IMGS_Special/20200604_doc_test \
/home/nas01/grp_IMRECOG/wangfei11/data_66/OnlineModel/scene_zhongkong_10/label_10.txt \
/home/nas01/grp_IMRECOG/wangfei11/data_66/OnlineModel/scene_zhongkong_10/20200609/result_score_36756_thres.txt


/home/work/wangfei11/slim_scene/Evaluate/Eval_pb_IMGS_in_folder.sh \
/home/nas01/grp_IMRECOG/TF_RECORD/AI_scene_central/TF_3_20200605/CKPT/CKPT_MV1QC_mixup_bs200_lrconf_1gpu_quant/model.ckpt-75051.quant.pb \
MobilenetV1/Predictions \
scene \
224 \
/home/work/wangfei11/data/AI_scene_back/Test_IMGS_Special/20200604_doc_test \
/home/nas01/grp_IMRECOG/wangfei11/data_66/OnlineModel/scene_zhongkong/label_3.txt \
/home/nas01/grp_IMRECOG/TF_RECORD/AI_scene_central/TF_3_20200605/CKPT/CKPT_MV1QC_mixup_bs200_lrconf_1gpu_quant/result_score_75051_thres.txt


/home/work/wangfei11/slim_scene/Evaluate/Eval_pb_IMGS_in_folder.sh \
/home/nas01/grp_IMRECOG/TF_RECORD/AI_scene_central/TF_3_20200605/CKPT/CKPT_MV1QC_mixup_bs200_lrconf_1gpu_quant/model.ckpt-75051.quant.pb \
MobilenetV1/Predictions \
scene \
224 \
/home/work/wangfei11/data/AI_scene_central/Test_IMGS_Docs/test_imgs_doc/other_scene \
/home/nas01/grp_IMRECOG/wangfei11/data_66/OnlineModel/scene_zhongkong/label_3.txt \
/home/nas01/grp_IMRECOG/TF_RECORD/AI_scene_central/TF_3_20200605/CKPT/CKPT_MV1QC_mixup_bs200_lrconf_1gpu_quant/result_score_75051_thres.txt

!

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64/

if [ $# -lt 7 ]
then
	echo "Usage: Eval_pb_IMGS_in_folder.sh pb_path output_node_names preprocess_name img_size EVAL_IMGS_DIR LABEL_FILE thres_file"
	exit
fi

pb_path=$1
output_node_names=$2
preprocess_name=$3
img_size=$4
EVAL_IMGS_DIR=$5
LABEL_FILE=$6
thres_file=$7
score_file=${EVAL_IMGS_DIR}/result_score.txt
cp ${thres_file} ${EVAL_IMGS_DIR}/result_score_thres.txt

LABEL_NUM=`echo ${LABEL_FILE} | awk -F"_" '{print $NF}' | awk -F"." '{print $1}'`


find ${EVAL_IMGS_DIR} -name "*.*" | egrep -i "jpg|jpeg|png|bmp" > ${EVAL_IMGS_DIR}/list.txt
label_idx_in_path=`echo $EVAL_IMGS_DIR | awk -F"/" '{print NF}'`
let label_idx_in_path+=1
awk -v label_idx_in_path=${label_idx_in_path} 'NR==FNR{A[$1]=$2} NR>FNR{n=split($1,a, "/"); if(a[label_idx_in_path] in A){print FNR"\t"$1"\t"A[a[label_idx_in_path]]}}' ${LABEL_FILE} ${EVAL_IMGS_DIR}/list.txt > ${EVAL_IMGS_DIR}/list_gt.txt

	
gpu=`nvidia-smi  | grep Default | awk '{print NR-1"\t"$11-$9-5000}' | grep -v '-' | sort -k2 -n | head -n1 | awk '{print $1}'`
while [ ! ${gpu} ]
do
	echo "no available gpu, will exit"
	exit
done


echo "---------------------gpu: " ${gpu}
CUDA_VISIBLE_DEVICES="${gpu}"   python -u /home/work/wangfei11/slim_scene/Evaluate/compute_scores.py \
	--pb_path=${pb_path} \
	--output_node_names=${output_node_names} \
	--image_gt_list=${EVAL_IMGS_DIR}/list_gt.txt \
	--img_root_path=/ \
	--preprocess_name=${preprocess_name} \
	--result_file=${score_file} \
	--img_size=${img_size}

	
python -u /home/work/wangfei11/slim_scene/Evaluate/getCrossMatrix.py ${score_file} \
	${EVAL_IMGS_DIR}/list_gt.txt ${LABEL_NUM} ${LABEL_FILE} \
	/home/work/wangfei11/data/AI_scene_back/Test_IMGS/Eval_label_map_restrict.txt

