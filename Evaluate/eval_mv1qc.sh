export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64/

if [ $# -lt 1 ]
then
    echo "Need LabelCount!"
    exit
fi

echo "nohup /home/work/wangfei11/slim_scene/Evaluate/Eval_ckpt_dir_base.sh mobilenet_v1_qc MobilenetV1/Predictions . scene 224 $1 `pwd` > evallog.txt 2>&1 &"

uname -a > evallog.txt

# 最后的参数 pwd 为了方便在ps时看在哪个路径启动的脚本，没有其他用途
nohup /home/work/wangfei11/slim_scene/Evaluate/Eval_ckpt_dir_base.sh mobilenet_v1_qc MobilenetV1/Predictions . scene 224 $1 `pwd` >> evallog.txt 2>&1 &
