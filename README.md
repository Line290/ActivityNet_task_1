# ActivityNet_task_1
This repository holds the codes for ActivityNet Challenge Task 1: Temporal Action Proposals   
# Related repositories:
[Temporal Segment Networks][TSN]  
[TSN PyTorch][TSN_pytorch]  
[BSN-boundary-sensitive-network.pytorch][BSN]  
[anet2016 CUHK][CUHK_anet]

# *Update
DPN92 RGB+Flow nonrescale feature [Baidu_pan_passwd_x8ff][dpn92]
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
# Ensemble
## Soft-NMS layer ensemble
Naive greed search to find a combination. 
```bash
   mv ./BSN/ensemble/* ./BSN
   # need to add
```
# Demo
Put TSN RGB trained model in the folder ``./demo_activity``  
Download:  
dpn92_RGB_k600_fold_2_rgb_model_best.pth.tar, [google_drive][google_drive] or [baidu_pan_passwd_38cw][baidu_pan]  
k600_dpn92_fold_0_flow_model_best.pth.tar: [google_drive][dpn92_flow_pretrained_model_google] or [baidu_pan_passwd_7k1j][dpn92_flow_pretrained_model_baidu]
```bash
cd demo_activity && sh run.sh VIDEO_PATH
```









[TSN]:https://github.com/yjxiong/temporal-segment-networks
[TSN_pytorch]:https://github.com/yjxiong/tsn-pytorch
[BSN]:https://github.com/wzmsltw/BSN-boundary-sensitive-network.pytorch#prerequisites
[CUHK_anet]:https://github.com/yjxiong/anet2016-cuhk
[Extract Frames and Optical Flow Images]:https://github.com/yjxiong/temporal-segment-networks#extract-frames-and-optical-flow-images
[google_drive]:https://drive.google.com/open?id=1014wTFhfv5Cr0vH49x4L4JACtgS8AXw2
[baidu_pan]:https://pan.baidu.com/s/1rDHf8g_uKC_qJReE1pDd2A
[dpn92]:https://pan.baidu.com/s/1pJoAY2MwfHiDFXtuQmiWqA
[dpn92_flow_pretrained_model_google]:https://drive.google.com/open?id=1wHXeek6QCxb5irOMoOI7Q2sdjTaQoe1M
[dpn92_flow_pretrained_model_baidu]:https://pan.baidu.com/s/1MdBen2WwHoJ5jcW1mKr15Q
