export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64/

if [ $# -lt 2 ]
then
    echo "Need LabelCount and CKPT!"
    exit
fi

# 最后的参数 pwd 为了方便在ps时看在哪个路径启动的脚本，没有其他用途
/home/work/wangfei11/slim_scene/Evaluate/Eval_ckpt_one_only_calc_matrix.sh mobilenet_v1_qc MobilenetV1/Predictions $2 scene 224 $1 `pwd`
