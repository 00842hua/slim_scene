sed 's#/home/work/wangfei11/data/screen_scene/Test_IMGS/IMGS/#G:/MI_Corpus/ScreenScene/Test_IMGS/IMGS/#' RESULT/*matrix_*badcase.txt > RESULT/badcases_win_path.txt

cat RESULT/badcases_win_path.txt | awk '{print $3}' | sort | uniq | xargs -i ./genNotPrecise.sh {}

cat RESULT/badcases_win_path.txt | awk '{print $2}' | sort | uniq | awk -F"#" '{for(i=1; i<=NF; i++){print $i}}' | sort | uniq | xargs -i ./genNotRecall.sh {}