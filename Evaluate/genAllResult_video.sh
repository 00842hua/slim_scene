sed 's#/home/work/wangfei11/data/video_labeling/Test_IMGS_SmallCate/#G:/MI_Corpus/VideoCorpus/VideoClassificationByImg/TEST_IMGS_SMALLCATE_Safe/#' RESULT/*matrix_*badcase.txt > RESULT/badcases_win_path.txt

cat RESULT/badcases_win_path.txt | awk '{print $3}' | sort | uniq | xargs -i ./genNotPrecise.sh {}

cat RESULT/badcases_win_path.txt | awk '{print $2}' | sort | uniq | awk -F"#" '{for(i=1; i<=NF; i++){print $i}}' | sort | uniq | xargs -i ./genNotRecall.sh {}