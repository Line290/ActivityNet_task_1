for i in 14 15 ;do
    CUDA_VISIBLE_DEVICES=7 nohup python extract_anet_feature.py --list_file="./lists_16/list.txt.new$i" --weights="./best_models/resnet152_result/fold1/k600_rgb_resnet152/_rgb_model_best.pth.tar" --arch="resnet152" --save_dir="./save_features/fold1/k600_rgb_resnet152" --b=256 --num_workers=4 --num_snippet=8 --modality="RGB" > ./extract_feature_log/log_$i.log &
done

#for i in 12 13 14 15 ;do
#    CUDA_VISIBLE_DEVICES=7 nohup python extract_anet_feature_flow.py "./lists_16/list.txt.new$i" "./best_models/resnet152_result/fold1/k600_flow_resnet152/_flow_model_best.pth.tar" --arch="resnet152" --save_dir="./save_features/fold1/k600_flow_resnet152" --b=48 --num_workers=2 --modality="Flow" > ./extract_feature_log/log_$i.log &
#done
