# ActivityNet_task_1
This repository holds the codes for ActivityNet Challenge Task 1: Temporal Action Proposals   
# Related repository:
[Temporal Segment Networks][TSN]  
[TSN PyTorch][TSN_pytorch]  
[BSN-boundary-sensitive-network.pytorch][BSN]  
[anet2016 CUHK][CUHK_anet]

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
### Rescaling RGB and Flow's features and concatenate them
```bash
cd ./BSN
python data_process.py arch ../TSN/RGB_feature_folder ../TSN/Flow_feature_folder num_works
```
### Training, evaluating and testing BSN
```bash
sh bsn_3fold_diff_backbone.sh
```
# Final result proposal
```bash
./BSN/output/result_proposal.json
```










[TSN]:https://github.com/yjxiong/temporal-segment-networks
[TSN_pytorch]:https://github.com/yjxiong/tsn-pytorch
[BSN]:https://github.com/wzmsltw/BSN-boundary-sensitive-network.pytorch#prerequisites
[CUHK_anet]:https://github.com/yjxiong/anet2016-cuhk
[Extract Frames and Optical Flow Images]:https://github.com/yjxiong/temporal-segment-networks#extract-frames-and-optical-flow-images