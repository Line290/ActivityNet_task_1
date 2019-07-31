#!/bin/bash
### feature path "k600_resnet152_fold_0_rescale", "k600_resnet152_fold_0_nonrescale"
#for backbone in 'k600_resnet152' 'k600_dpn92'
# for backbone in 'se101'
for backbone in 'cuhk'
do
#     for fold in 0 1 2
    for fold in 2
    do
        arch="k600_${backbone}_fold_${fold}_norm"
#         feat_path="./data/activitynet_feature_cuhk/${backbone}_k600_3fold_feat/k600_${backbone}_fold_${fold}_rescale"
        feat_path="./data/activitynet_feature_cuhk/rescale_100/csv_mean_100"
        video_info="./data/activitynet_annotations/3fold_csv/fold_${fold}_video_info.csv"
        eval_json="./data/activitynet_annotations/3fold_csv/fold_${fold}_activity_net.v1-3.min.json"
        subopts="--feature_path ${feat_path} --video_info ${video_info} --arch ${arch} --fix_scale rescale"
        
        python main.py --module TEM --mode train ${subopts}

        python main.py --module TEM --mode inference ${subopts}

        python main.py --module PGM ${subopts}
        python main.py --module PEM --mode train ${subopts}

        python main.py --module PEM --mode inference --pem_batch_size 1 ${subopts}

        python main.py --module Post_processing ${subopts}

        python main.py --module Evaluation ${subopts} --eval_json ${eval_json}
        ##test
        python main.py --module PEM --mode inference --pem_batch_size 1 --pem_inference_subset testing ${subopts}
        python main.py --module Post_processing --pem_inference_subset testing ${subopts}
        
#         feat_path_nonrescale="./data/activitynet_feature_cuhk/${backbone}_k600_3fold_feat/k600_${backbone}_fold_${fold}_nonrescale"
        feat_path_nonrescale="./data/activitynet_feature_cuhk/non_rescale"
        subopts_nonrescale="--feature_path ${feat_path_nonrescale} --video_info ${video_info} --arch ${arch} --fix_scale nonrescale"
        ######### nonrescale ####
        python main.py --module TEM --mode inference --tem_batch_size 1 ${subopts_nonrescale}

        python main.py --module PGM ${subopts_nonrescale}

        python main.py --module PEM --mode inference --pem_batch_size 1 ${subopts_nonrescale}

        python main.py --module Post_processing ${subopts_nonrescale}

        python main.py --module Evaluation ${subopts_nonrescale} --eval_json ${eval_json}
        ###test
        python main.py --module PEM --mode inference --pem_batch_size 1 --pem_inference_subset testing ${subopts_nonrescale}
        python main.py --module Post_processing --pem_inference_subset testing ${subopts_nonrescale}
    done
done