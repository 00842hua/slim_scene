
scene=$1
resfile=$2

target_dir=RESULT/NOT_RECALL_HTML

if [ ! -d ${target_dir} ]
then
    mkdir -p ${target_dir}
fi

target_html=${target_dir}/${scene}.html


cat ${resfile} | sed 's#/storage/emulated/0/#../#' | sed 's#\##/#' | sed 's#000_all_4#000#' | sed 's#000_all_1#000#'| sed 's#000_all_2#000#'| sed 's#000_all_3#000#'| sed 's#000_all#000#' | awk -F"|" -v scene=${scene} '$2==scene{if($2!=$3){print}}'  | awk -F"|" -v labelidx=3 -f show_result_html.awk > ${target_html}
