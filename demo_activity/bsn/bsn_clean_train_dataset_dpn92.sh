#!/bin/bash
### feature path "k600_resnet152_fold_0_rescale", "k600_resnet152_fold_0_nonrescale"
#for backbone in 'k600_resnet152' 'k600_dpn92'
backbone='dpn92'
fold=2
arch="k600_${backbone}_fold_${fold}_200"
video_info="../video_info.csv"
video_anno="../video_anno.json"
result_file="./output/result_proposal.json"
feat_path_nonrescale="../output_ftrs_RGB_dpn92"
subopts_nonrescale="--feature_path ${feat_path_nonrescale} --video_info ${video_info} --video_anno ${video_anno} --arch ${arch} --fix_scale nonrescale"
######### nonrescale ####
python main.py --module TEM --mode inference --tem_batch_size 1 ${subopts_nonrescale}

python main.py --module PGM ${subopts_nonrescale}

python main.py --module PEM --mode inference --pem_batch_size 1 ${subopts_nonrescale}

python main.py --module Post_processing ${subopts_nonrescale} --result_file ${result_file}
