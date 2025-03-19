sed 's#/home/nas01/grp_IMRECOG/wangfei11/data_66/AI_Zhedang#G:/MI_Corpus/AI_Zhedang_duanshipin/#' RESULT/*matrix_*badcase.txt | sed 's#/home/work/wangfei11/data/AI_Zhedang_duanshipin#G:/MI_Corpus/AI_Zhedang_duanshipin/#' RESULT/*matrix_*badcase.txt > RESULT/badcases_win_path.txt

cat RESULT/badcases_win_path.txt | awk '{print $3}' | sort | uniq | xargs -i ./genNotPrecise.sh {}

cat RESULT/badcases_win_path.txt | awk '{print $2}' | sort | uniq | awk -F"#" '{for(i=1; i<=NF; i++){print $i}}' | sort | uniq | xargs -i ./genNotRecall.sh {}
