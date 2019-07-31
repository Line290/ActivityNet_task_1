#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python  main.py activitynet RGB train_pos_list.txt val_pos_list.txt  --output_dir="./log/20190518_rgb_pose256_resnet152_pos2/" -j 32 -b 24 --num_segments=20
#--pre-model="./best_models/pose256_resnet152/_rgb_model_best.pth.tar"

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python  main_base.py activitynet Flow train_list5.txt val_list5.txt  --output_dir="./log/20190520_flow_pose256_kinetics/" -j 32 -b 64 --num_segments=7

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python  main.py kinetics Flow ../kinetics_600/list/train_list.txt ../kinetics_600/list/val_list.txt  --output_dir="./log/20190605_k600_flow_inceptionv4/" -j 16 -b 128 --num_segments=7
#--resume="./best_models/pretrain_models/k600_flow_se101/_flow_model_best.pth.tar"
