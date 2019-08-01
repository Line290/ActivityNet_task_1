#!/bin/bash
# conda env
#/users/lindq/.conda/envs/pytoch_py2

# tsn inference
#conda activate py27_pytorch0.3.1

CUDA_VISIBLE_DEVICES=3 python extract_anet_feature.py $1 # video_path

# bsn inference
#conda activate pytoch_py2
cd bsn
CUDA_VISIBLE_DEVICES=3 sh bsn_clean_train_dataset_dpn92.sh

# process results
cd ..
# video_label_name_path, logits_path, res_path, video_info, save_path
python result_process.py ./bsn/data/activitynet_annotations/action_name.csv ./output_ftrs_RGB_dpn92 ./bsn/output/result_proposal.json ./video_info.csv ./final_result.json
