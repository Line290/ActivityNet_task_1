# ActivityNet_task_1
This repository holds the codes for ActivityNet Challenge Task 1: video action proposal

# Usage Guide
## Setting
```
conda env create -f environment.yml
```
## Data Preparation
[Extract Frames and Optical Flow Images][Extract Frames and Optical Flow Images]
## TSN
### Training TSN
RGB&Flow
```bash
cd ./TSN && sh run.sh
```
### Extracting video's feature
RGB&Flow
```bash
sh extract_feature.sh
```
## BSN
### Scaling RGB and Flow features and concatenate
```bash
cd ./BSN
python data_process.py arch ../TSN/RGB_feature_folder ../TSN/Flow_feature_folder num_works
```
### Training evaluating and testing BSN
```bash
sh bsn_3fold_diff_backbone.sh
```
# Final result proposal
```bash
./BSN/output/result_proposal.json
```











[Extract Frames and Optical Flow Images]:https://github.com/yjxiong/temporal-segment-networks#extract-frames-and-optical-flow-images