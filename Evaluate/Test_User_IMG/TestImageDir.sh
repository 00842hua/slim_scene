
# 对用户图验证模型的准确率，主要是验证类型没有覆盖的图片会被分到什么类型

:<<!
使用方法如下

------ 1、调用本脚本，对用户图进行predict，并生成对应的html文件
/home/work/wangfei11/slim_scene/Evaluate/Test_User_IMG/TestImageDir.sh \
/home/work/wangfei11/data/AI_Stove/TF_5_20200605/CKPT/CKPT_MV1QC_mixup_bs200_lrconf_1gpu_quant_v100/model.ckpt-52076.quant.pb \
MobilenetV1/Predictions_sigmoid \
/home/work/wangfei11/data/AI_Stove/labels_5.txt \
scene_new \
1 \
/home/work/wangfei11/data/AI_Stove/Test_IMGS



------ 2、在根目录启动python的http服务     cd / ; python -m SimpleHTTPServer  1204

------ 3、在浏览器里输入如下地址访问，注意目录是pb路径后加_html_result。需要vpn并且使用服务器代理 proxy.pt.xiaomi.com:80

http://tj-hadoop-dl89.kscn:1204/home/work/wangfei11/data/AI_Stove/TF_5_20200605/CKPT/CKPT_MV1QC_mixup_bs200_lrconf_1gpu_quant_v100/model.ckpt-52076.quant.pb_html_result

!


if [ $# -lt 6 ]
then
	echo "Usage $0 pb_path outputnode_name label_file_path preprocess_type(scene_new or inception) gpu_idx img_dir"
	exit
fi

PB_PATH=$1
OUTPUTNODE=$2
LABE_FILE=$3
PREPROCESS=$4
GPU_IDX=$5
IMG_DIR=$6
find ${IMG_DIR} -name "*.*" | egrep -i "jpg|jpeg|png|bmp" > ${IMG_DIR}/test_list.txt
IMG_LIST=${IMG_DIR}/test_list.txt
RESULT_FILE=${PB_PATH}.result.txt
HTML_RESULT_DIR=${PB_PATH}_html_result


CUDA_VISIBLE_DEVICES=$GPU_IDX \
python /home/work/wangfei11/slim_scene/predict_batch_pb.py \
--pb_path=$PB_PATH \
--output_node_names=$OUTPUTNODE \
--preprocess_type=$PREPROCESS \
--image_size=224 \
--batch_size=100 \
--labels_file=$LABE_FILE \
--image_list=$IMG_LIST \
--result_file=$RESULT_FILE


rm -rf $HTML_RESULT_DIR
mkdir $HTML_RESULT_DIR

while read line
do
	label=`echo $line|cut -d":" -f2`
	#echo $line
	echo $label
	cat $RESULT_FILE | awk -v label=$label '{if ($NF==label && $(NF-1) > 0.1){print $0}}' | awk -f /home/work/wangfei11/slim_scene/Evaluate/Test_User_IMG/show_result_html.awk | sed 's/#/%23/g' > ${HTML_RESULT_DIR}/${label}.html
done < $LABE_FILE
