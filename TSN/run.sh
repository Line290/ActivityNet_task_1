# RGB
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py activitynet RGB train_list.txt val_list.txt  -j 48 -b 80 --arch dpn92 --output_dir ./logs/20190513_dpn92_RGB/ --snapshot_pref dpn92_RGB

# Flow
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py activitynet Flow train_list.txt val_list.txt  -j 48 -b 80 --arch dpn92 --dropout 0.7 --lr_steps 190 300 --epochs 340 --gd 20 --snapshot_pref dpn92_flow
