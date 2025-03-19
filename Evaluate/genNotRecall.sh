badcase_file=RESULT/badcases_win_path.txt

scene=$1

target_dir=RESULT/NOT_RECALL_HTML

if [ ! -d ${target_dir} ]
then
    mkdir -p ${target_dir}
fi

target_html=${target_dir}/${scene}.html

awk -v scene=${scene} '{n=split($2, a, "#"); for (i=1; i<=n; i++){ if(a[i]==scene){print; break;}}}' ${badcase_file} | sed 's/#/%23/' | awk -f show_result_html.awk > ${target_html}