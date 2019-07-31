# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pandas
import numpy
import json
import torch.utils.data as data
import os
import torch

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

class VideoDataSet(data.Dataset):
    def __init__(self,opt,subset="train"):
        self.temporal_scale = opt["temporal_scale"]
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = opt["mode"]
        self.feature_path = opt["feature_path"]
        self.boundary_ratio = opt["boundary_ratio"]
        self.video_info_path = opt["video_info"]
        self.video_anno_path = opt["video_anno"]
        self.fix_scale = opt['fix_scale']
        self._getDatasetDict()
        
    def _getDatasetDict(self):
        anno_df = pd.read_csv(self.video_info_path)
        anno_database= load_json(self.video_anno_path)
        self.video_dict = {}
        for i in range(len(anno_df)):
            video_name=anno_df.video.values[i]
            video_info=anno_database[video_name]
            video_subset=anno_df.subset.values[i]
            # add by lindq
            if self.subset == 'trainval':
                if video_subset != 'testing':
                    self.video_dict[video_name] = video_info
            # end add
            if self.subset == "full":
                self.video_dict[video_name] = video_info
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info
        self.video_list = self.video_dict.keys()
        print "%s subset video numbers: %d" %(self.subset,len(self.video_list))

    def __getitem__(self, index):
        video_data,anchor_xmin,anchor_xmax = self._get_base_data(index)
        if self.mode == "train":
            match_score_action,match_score_start,match_score_end =  self._get_train_label(index,anchor_xmin,anchor_xmax)
            return video_data,match_score_action,match_score_start,match_score_end
        else:
            return index,video_data,anchor_xmin,anchor_xmax
        
    def _get_base_data(self,index):
        video_name=self.video_list[index]
        anchor_xmin=[self.temporal_gap*i for i in range(self.temporal_scale)]
        anchor_xmax=[self.temporal_gap*i for i in range(1,self.temporal_scale+1)]
#         if self.fix_scale:
# #             video_df=pd.read_csv(self.feature_path+ "csv_mean_"+str(self.temporal_scale)+"/"+video_name+".csv")
#             video_df=pd.read_csv(os.path.join(self.feature_path, video_name+".csv"))
#         else:
# #             video_df=pd.read_csv(self.feature_path+ "non_rescale"+"/"+video_name+".csv")
        video_df=pd.read_csv(os.path.join(self.feature_path, video_name+".csv"))
        video_data = video_df.values[:,:200]
        # applied normalized
        if False:
            length, dim = video_data.shape
            if dim == 400:
#                 rgb_mean = video_data[:, :200].mean(axis=1, keepdims=True)
#                 rgb_var = video_data[:, :200].var(axis=1, keepdims=True)
#                 flow_mean = video_data[:, 200:].mean(axis=1, keepdims=True)
#                 flow_var = video_data[:, 200:].var(axis=1, keepdims=True)
#                 rgb_norm = ( video_data[:, :200] - rgb_mean) / (rgb_var + 1e-10)
#                 flow_norm = (video_data[:, 200:] - flow_mean) / (flow_var + 1e-10)
                rgb_norm = video_data[:, :200]/(np.linalg.norm(video_data[:, :200], axis=1, keepdims=True)+1e-10)
                flow_norm =  video_data[:, 200:]/(np.linalg.norm(video_data[:, 200:], axis=1, keepdims=True)+1e-10)
                video_data = np.hstack((rgb_norm, flow_norm))
            else:
                video_data = (video_data - video_data.mean(axis=1, keepdims=True)) / (video_data.var(axis=1, keepdims=True) + 1e-10)
        # end normalized
        video_data = torch.Tensor(video_data)
        video_data = torch.transpose(video_data,0,1)
        video_data.float()
        return video_data,anchor_xmin,anchor_xmax
    
    def _get_train_label(self,index,anchor_xmin,anchor_xmax): 
        video_name=self.video_list[index]
        video_info=self.video_dict[video_name]
        video_frame=video_info['duration_frame']
        video_second=video_info['duration_second']
        feature_frame=video_info['feature_frame']
        corrected_second=float(feature_frame)/video_frame*video_second
        video_labels=video_info['annotations']
    
        gt_bbox = []
        for j in range(len(video_labels)):
            tmp_info=video_labels[j]
            tmp_start=max(min(1,tmp_info['segment'][0]/corrected_second),0)
            tmp_end=max(min(1,tmp_info['segment'][1]/corrected_second),0)
            gt_bbox.append([tmp_start,tmp_end])
            
        gt_bbox=np.array(gt_bbox)
        gt_xmins=gt_bbox[:,0]
        gt_xmaxs=gt_bbox[:,1]

        gt_lens=gt_xmaxs-gt_xmins
        gt_len_small=np.maximum(self.temporal_gap,self.boundary_ratio*gt_lens)
        gt_start_bboxs=np.stack((gt_xmins-gt_len_small/2,gt_xmins+gt_len_small/2),axis=1)
        gt_end_bboxs=np.stack((gt_xmaxs-gt_len_small/2,gt_xmaxs+gt_len_small/2),axis=1)
        
        match_score_action=[]
        for jdx in range(len(anchor_xmin)):
            match_score_action.append(np.max(self._ioa_with_anchors(anchor_xmin[jdx],anchor_xmax[jdx],gt_xmins,gt_xmaxs)))
        match_score_start=[]
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(self._ioa_with_anchors(anchor_xmin[jdx],anchor_xmax[jdx],gt_start_bboxs[:,0],gt_start_bboxs[:,1])))
        match_score_end=[]
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(self._ioa_with_anchors(anchor_xmin[jdx],anchor_xmax[jdx],gt_end_bboxs[:,0],gt_end_bboxs[:,1])))
        match_score_action = torch.Tensor(match_score_action)
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        return match_score_action,match_score_start,match_score_end

    def _ioa_with_anchors(self,anchors_min,anchors_max,box_min,box_max):
        len_anchors=anchors_max-anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        scores = np.divide(inter_len, len_anchors)
        return scores
    
    def __len__(self):
        return len(self.video_list)


class ProposalDataSet(data.Dataset):
    def __init__(self,opt,subset="train"):
        
        self.subset=subset
        self.mode = opt["mode"]
        if self.mode == "train":
            self.top_K = opt["pem_top_K"]
        else:
            self.top_K = opt["pem_top_K_inference"]
        self.video_info_path = opt["video_info"]
        self.video_anno_path = opt["video_anno"]
        self.opt = opt
        self._getDatasetDict()
        
    def _getDatasetDict(self):
        anno_df = pd.read_csv(self.video_info_path)
        anno_database= load_json(self.video_anno_path)
        self.video_dict = {}
        for i in range(len(anno_df)):
            video_name=anno_df.video.values[i]
            video_info=anno_database[video_name]
            video_subset=anno_df.subset.values[i]
            # add by lindq
            if self.subset == 'trainval':
                if video_subset != 'testing':
                    self.video_dict[video_name] = video_info
            # end add
            if self.subset == "full":
                self.video_dict[video_name] = video_info
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info
        self.video_list = self.video_dict.keys()
        print "%s subset video numbers: %d" %(self.subset,len(self.video_list))

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        video_name = self.video_list[index]
        pdf=pandas.read_csv("./output/"+self.opt["arch"]+self.opt["fix_scale"]+"_PGM_proposals/"+video_name+".csv")
        pdf=pdf[:self.top_K]
        video_feature = numpy.load("./output/"+self.opt["arch"]+self.opt["fix_scale"]+"_PGM_feature/" + video_name+".npy")
        video_feature = video_feature[:self.top_K,:]
        #print len(video_feature),len(pdf)
        video_feature = torch.Tensor(video_feature)

        if self.mode == "train":
            video_match_iou = torch.Tensor(pdf.match_iou.values[:])
            # append
            video_is_whole_length = torch.Tensor(pdf.is_whole_length.values[:])
            return video_feature,[video_match_iou, video_is_whole_length]
            # finished append
#             return video_feature,video_match_iou
        else:
            video_xmin =pdf.xmin.values[:]
            video_xmax =pdf.xmax.values[:]
            video_xmin_score = pdf.xmin_score.values[:]
            video_xmax_score = pdf.xmax_score.values[:]
            return video_feature,video_xmin,video_xmax,video_xmin_score,video_xmax_score
        