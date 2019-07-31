# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import sys
sys.dont_write_bytecode = True
import os
import json
import torch
import torchvision
import torch.nn.parallel
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
import opts
from dataset import VideoDataSet,ProposalDataSet
from models import TEM,PEM
from loss_function import TEM_loss_function,PEM_loss_function
import pandas as pd
from pgm import PGM_proposal_generation,PGM_feature_generation
from post_processing_fusion_search import BSN_post_processing
from eval_search import evaluation_proposal
from load_all_PEM_result import load_all_PEM_results
GPU_IDs = [0]
def train_TEM(data_loader,model,optimizer,epoch,writer,opt):
    model.train()
    epoch_action_loss = 0
    epoch_start_loss = 0
    epoch_end_loss = 0
    epoch_cost = 0
    for n_iter,(input_data,label_action,label_start,label_end) in enumerate(data_loader):
        TEM_output = model(input_data)
        loss = TEM_loss_function(label_action,label_start,label_end,TEM_output,opt)
        cost = loss["cost"] 
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        epoch_action_loss += loss["loss_action"].cpu().detach().numpy()
        epoch_start_loss += loss["loss_start"].cpu().detach().numpy()
        epoch_end_loss += loss["loss_end"].cpu().detach().numpy()
        epoch_cost += loss["cost"].cpu().detach().numpy()
    
    writer.add_scalars('data/action', {'train': epoch_action_loss/(n_iter+1)}, epoch)
    writer.add_scalars('data/start', {'train': epoch_start_loss/(n_iter+1)}, epoch)
    writer.add_scalars('data/end', {'train': epoch_end_loss/(n_iter+1)}, epoch)
    writer.add_scalars('data/cost', {'train': epoch_cost/(n_iter+1)}, epoch)

    print "TEM training loss(epoch %d): action - %.03f, start - %.03f, end - %.03f" %(epoch,epoch_action_loss/(n_iter+1),
                                                                                        epoch_start_loss/(n_iter+1),
                                                                                        epoch_end_loss/(n_iter+1))

def test_TEM(data_loader,model,epoch,writer,opt):
    model.eval()
    epoch_action_loss = 0
    epoch_start_loss = 0
    epoch_end_loss = 0
    epoch_cost = 0
    for n_iter,(input_data,label_action,label_start,label_end) in enumerate(data_loader):
        
        TEM_output = model(input_data)
        loss = TEM_loss_function(label_action,label_start,label_end,TEM_output,opt)
        epoch_action_loss += loss["loss_action"].cpu().detach().numpy()
        epoch_start_loss += loss["loss_start"].cpu().detach().numpy()
        epoch_end_loss += loss["loss_end"].cpu().detach().numpy()
        epoch_cost += loss["cost"].cpu().detach().numpy()
    
    writer.add_scalars('data/action', {'test': epoch_action_loss/(n_iter+1)}, epoch)
    writer.add_scalars('data/start', {'test': epoch_start_loss/(n_iter+1)}, epoch)
    writer.add_scalars('data/end', {'test': epoch_end_loss/(n_iter+1)}, epoch)
    writer.add_scalars('data/cost', {'test': epoch_cost/(n_iter+1)}, epoch)
    
    print "TEM testing  loss(epoch %d): action - %.03f, start - %.03f, end - %.03f" %(epoch,epoch_action_loss/(n_iter+1),
                                                                                        epoch_start_loss/(n_iter+1),
                                                                                        epoch_end_loss/(n_iter+1))
    state = {'epoch': epoch + 1,
                'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"]+"/"+opt["arch"]+"_tem_checkpoint.pth.tar" )
    if epoch_cost< model.module.tem_best_loss:
        model.module.tem_best_loss = np.mean(epoch_cost)
        torch.save(state, opt["checkpoint_path"]+"/"+opt["arch"]+"_tem_best.pth.tar" )

def train_PEM(data_loader,model,optimizer,epoch,writer,opt):
    model.train()
    epoch_iou_loss = 0
    
    for n_iter,(input_data,label_iou,is_whole_lenght) in enumerate(data_loader):
        PEM_output = model(input_data)
        iou_loss = PEM_loss_function(PEM_output,[label_iou, is_whole_lenght],model,opt)
        optimizer.zero_grad()
        iou_loss.backward()
        optimizer.step()
        epoch_iou_loss += iou_loss.cpu().detach().numpy()

    writer.add_scalars('data/iou_loss', {'train': epoch_iou_loss/(n_iter+1)}, epoch)
    
    print "PEM training loss(epoch %d): iou - %.04f" %(epoch,epoch_iou_loss/(n_iter+1))

def test_PEM(data_loader,model,epoch,writer,opt):
    model.eval()
    epoch_iou_loss = 0
    
    for n_iter,(input_data,label_iou,is_whole_lenght) in enumerate(data_loader):
        PEM_output = model(input_data)
        iou_loss = PEM_loss_function(PEM_output,[label_iou,is_whole_lenght],model,opt)
        epoch_iou_loss += iou_loss.cpu().detach().numpy()

    writer.add_scalars('data/iou_loss', {'validation': epoch_iou_loss/(n_iter+1)}, epoch)
    
    print ("PEM testing  loss(epoch %d): iou - %.04f" %(epoch,epoch_iou_loss/(n_iter+1)))
    
    state = {'epoch': epoch + 1,
                'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"]+"/"+opt["arch"]+"_pem_checkpoint.pth.tar" )
    if epoch_iou_loss<model.module.pem_best_loss :
        model.module.pem_best_loss = np.mean(epoch_iou_loss)
        torch.save(state, opt["checkpoint_path"]+"/"+opt["arch"]+"_pem_best.pth.tar" )


def BSN_Train_TEM(opt):
    writer = SummaryWriter()
    model = TEM(opt)
    model = torch.nn.DataParallel(model, device_ids=GPU_IDs).cuda()
    
    optimizer = optim.Adam(model.parameters(),lr=opt["tem_training_lr"],weight_decay = opt["tem_weight_decay"])
    
    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt,subset="train"),
                                                batch_size=model.module.batch_size, shuffle=True,
                                                num_workers=8, pin_memory=True,drop_last=True)            
    
    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt,subset="validation"),
                                                batch_size=model.module.batch_size, shuffle=False,
                                                num_workers=8, pin_memory=True,drop_last=True)
        
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = opt["tem_step_size"], gamma = opt["tem_step_gamma"])
        
    for epoch in range(opt["tem_epoch"]):
        scheduler.step()
        train_TEM(train_loader,model,optimizer,epoch,writer,opt)
        test_TEM(test_loader,model,epoch,writer,opt)
    writer.close()
    


def BSN_Train_PEM(opt):
    writer = SummaryWriter()
    model = PEM(opt)
    model = torch.nn.DataParallel(model, device_ids=GPU_IDs).cuda()
    
    optimizer = optim.Adam(model.parameters(),lr=opt["pem_training_lr"],weight_decay = opt["pem_weight_decay"])
    
    def collate_fn(batch):
        batch_data = torch.cat([x[0] for x in batch])
        batch_iou = torch.cat([x[1][0] for x in batch])
        batch_is_whole_lenght = torch.cat([x[1][1] for x in batch])
        return batch_data,batch_iou, batch_is_whole_lenght
    
    train_loader = torch.utils.data.DataLoader(ProposalDataSet(opt,subset="train"),
                                                batch_size=model.module.batch_size, shuffle=True,
                                                num_workers=8, pin_memory=True,drop_last=True,collate_fn=collate_fn)            
    
    test_loader = torch.utils.data.DataLoader(ProposalDataSet(opt,subset="validation"),
                                                batch_size=model.module.batch_size, shuffle=True,
                                                num_workers=8, pin_memory=True,drop_last=True,collate_fn=collate_fn)
        
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = opt["pem_step_size"], gamma = opt["pem_step_gamma"])
        
    for epoch in range(opt["pem_epoch"]):
        scheduler.step()
        train_PEM(train_loader,model,optimizer,epoch,writer,opt)
        test_PEM(test_loader,model,epoch,writer,opt)
        
    writer.close()


def BSN_inference_TEM(opt):
    model = TEM(opt)
    checkpoint = torch.load(opt["checkpoint_path"]+"/"+opt["arch"]+"_tem_best.pth.tar")
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(base_dict)
    model = torch.nn.DataParallel(model, device_ids=GPU_IDs).cuda()
    model.eval()
    
    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt,subset="full"),
                                                batch_size=model.module.batch_size, shuffle=False,
                                                num_workers=8, pin_memory=True,drop_last=False)
#     test_loader = torch.utils.data.DataLoader(VideoDataSet(opt,subset="trainval"),
#                                                 batch_size=model.module.batch_size, shuffle=False,
#                                                 num_workers=8, pin_memory=True,drop_last=False)    
    columns=["action","start","end","xmin","xmax"]
    count = 0
    for index_list,input_data,anchor_xmin,anchor_xmax in test_loader:
        #for video with different length
#         if opt['fix_scale'] is False:
        if opt['fix_scale'] == 'nonrescale':
            if len(anchor_xmin) != input_data.shape[2]:
                temporal_scale = input_data.shape[2]
                temporal_gap = 1. / temporal_scale
                anchor_xmin=[temporal_gap*i for i in range(temporal_scale)]
                anchor_xmin = [torch.tensor([x]) for x in anchor_xmin]
                anchor_xmax=[temporal_gap*i for i in range(1,temporal_scale+1)]
                anchor_xmax = [torch.tensor([x]) for x in anchor_xmax]

        #############################################################
        TEM_output = model(input_data).detach().cpu().numpy()
        batch_action = TEM_output[:,0,:]
        batch_start = TEM_output[:,1,:]
        batch_end = TEM_output[:,2,:]
        
        index_list = index_list.numpy()
        anchor_xmin = np.array([x.numpy()[0] for x in anchor_xmin])
        anchor_xmax = np.array([x.numpy()[0] for x in anchor_xmax])
        
        for batch_idx,full_idx in enumerate(index_list):            
            video = test_loader.dataset.video_list[full_idx]
            video_action = batch_action[batch_idx]
            video_start = batch_start[batch_idx]
            video_end = batch_end[batch_idx]    
            video_result = np.stack((video_action,video_start,video_end,anchor_xmin,anchor_xmax),axis=1)
            video_df = pd.DataFrame(video_result,columns=columns)  
            video_df.to_csv("./output/"+opt["arch"]+opt["fix_scale"]+"_TEM_results/"+video+".csv",index=False)
            count += 1
        if count % 100 == 0:
            print('finish', count)
            sys.stdout.flush()
def BSN_inference_PEM(opt):
    model = PEM(opt)
    checkpoint = torch.load(opt["checkpoint_path"]+"/"+opt["arch"]+"_pem_best.pth.tar")
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(base_dict)
    model = torch.nn.DataParallel(model, device_ids=GPU_IDs).cuda()
    model.eval()
    
    test_loader = torch.utils.data.DataLoader(ProposalDataSet(opt,subset=opt["pem_inference_subset"]),
                                                batch_size=model.module.batch_size, shuffle=False,
                                                num_workers=8, pin_memory=True,drop_last=False)
    
    for idx,(video_feature,video_xmin,video_xmax,video_xmin_score,video_xmax_score) in enumerate(test_loader):
        video_name = test_loader.dataset.video_list[idx]
        video_conf = model(video_feature).view(-1).detach().cpu().numpy()
        video_xmin = video_xmin.view(-1).cpu().numpy()
        video_xmax = video_xmax.view(-1).cpu().numpy()
        video_xmin_score = video_xmin_score.view(-1).cpu().numpy()
        video_xmax_score = video_xmax_score.view(-1).cpu().numpy()
        
        df=pd.DataFrame()
        df["xmin"]=video_xmin
        df["xmax"]=video_xmax
        df["xmin_score"]=video_xmin_score
        df["xmax_score"]=video_xmax_score
        df["iou_score"]=video_conf
        
        df.to_csv("./output/"+opt["arch"]+opt["fix_scale"]+"_PEM_results/"+video_name+".csv",index=False)


def main(opt):
    if opt["module"] == "TEM":
        if opt["mode"] == "train":
            print "TEM training start"  
            BSN_Train_TEM(opt)
            print "TEM training finished" 
        elif opt["mode"] == "inference":
            print "TEM inference start"
            if not os.path.exists("output/"+opt["arch"]+opt["fix_scale"]+"_TEM_results"):
                os.makedirs("output/"+opt["arch"]+opt["fix_scale"]+"_TEM_results") 
            BSN_inference_TEM(opt)
            print "TEM inference finished"
        else:
            print "Wrong mode. TEM has two modes: train and inference"
          
    elif opt["module"] == "PGM":
        if not os.path.exists("output/"+opt["arch"]+opt["fix_scale"]+"_PGM_proposals"):
            os.makedirs("output/"+opt["arch"]+opt["fix_scale"]+"_PGM_proposals")
        print "PGM: start generate proposals"
        PGM_proposal_generation(opt)
        print "PGM: finish generate proposals"
        
        if not os.path.exists("output/"+opt["arch"]+opt["fix_scale"]+"_PGM_feature"):
            os.makedirs("output/"+opt["arch"]+opt["fix_scale"]+"_PGM_feature") 
        print "PGM: start generate BSP feature"
        PGM_feature_generation(opt)
        print "PGM: finish generate BSP feature"
    
    elif opt["module"] == "PEM":
        if opt["mode"] == "train":
            print "PEM training start"  
            BSN_Train_PEM(opt)
            print "PEM training finished"  
        elif opt["mode"] == "inference":
            if not os.path.exists("output/"+opt["arch"]+opt["fix_scale"]+"_PEM_results"):
                os.makedirs("output/"+opt["arch"]+opt["fix_scale"]+"_PEM_results") 
            print "PEM inference start"  
            BSN_inference_PEM(opt)
            print "PEM inference finished"
        else:
            print "Wrong mode. PEM has two modes: train and inference"
    
    elif opt["module"] == "Post_processing":
        all_PEM_results, num_fold, fold_path_mp = load_all_PEM_results()
        print("Success load all PEM results")
        max_auc = 0
        max_state = None
        #all_auc = [67.8940031356, 67.8325124118, 67.6377253724, 67.7577933107, 68.0293114711,67.9902207996,67.2231382284, 67.3738764045,67.7574274889,67.5690357983,67.7516788607,67.7326691926,67.9899007055,68.0310948524,67.9693950875,68.0369610661,67.8679775281, 67.3864841913,67.4476417559,67.7652861249,67.9869806637,67.6981186308,67.9510517377,67.9937548994,68.20611445,68.1691468513,67.3367977528,66.8307812908,65.9401685393,66.3082571205,67.641625294]
        #for i in range(31, num_fold):
        #    print(i)
        #    tmp_state = [-1]*num_fold
        #    tmp_state[i] = 1
        #    BSN_post_processing(opt, all_PEM_results, tmp_state)
        #    tmp_auc = evaluation_proposal(opt)
        #    all_auc.append(tmp_auc)
        #all_auc = np.array(all_auc)
        #top_idx = np.argsort(all_auc)[::-1]
        #print(all_auc)
        # init state
        #state = [-1]*num_fold
        state = [-1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1,-1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1]
        state[30] = -1
        BSN_post_processing(opt, all_PEM_results, state)
        pre_auc = evaluation_proposal(opt)
        #pre_auc = all_auc[top_idx[0]]
        max_auc = pre_auc
        max_state = state[:]
        for i in range(0, num_fold):
            state[i] = 1
            BSN_post_processing(opt, all_PEM_results, state)
            pre_auc = evaluation_proposal(opt)
            if pre_auc > max_auc:
                max_auc = pre_auc
                max_state = state[:]
                print("max auc: ", max_auc)
                print("max state: ", max_state)
            else:
                state[i] = -1
#         state[:10] = [1]*10
# #         state[1] = 1
#         pre_state = state[:]
#         BSN_post_processing(opt, all_PEM_results, pre_state)
#         pre_auc = evaluation_proposal(opt)
#         print("pre auc: ", pre_auc)
#         # climb mountain
#         for step in range(100):
#             for i in range(0, num_fold):
#                 state[i] *= -1
#     #             print "Post processing start"
#                 BSN_post_processing(opt, all_PEM_results, state)
#     #             print "Post processing finished"

#                 tmp_auc = evaluation_proposal(opt)
#                 print('temp auc: ',tmp_auc)
#                 print('state:', state)
#                 if tmp_auc>pre_auc:
#                     pre_state = state[:]
#                     pre_auc = tmp_auc
#                 # reset state
#                 state = pre_state[:]
     
# #         # climb mountain
# #         for step in range(100):
# #             for i in range(num_fold):
# #                 state[i] *= -1
        print("Final result:")
        print("max_val_auc: ", max_auc)
        print("All choose fold_path: ", max_state)
        for i in max_state:
            if i == 1:
                print(fold_path_mp[i])
        # save test json
        #opts["pem_inference_subset"] = "testing"
        #opts["result_file"] = "./final_test_result_val_"+str(max_auc)+".json"
        #BSN_post_processing(opt, all_PEM_results, max_state)
if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"]) 
    opt_file=open(opt["checkpoint_path"]+"/"+opt["arch"]+opt["fix_scale"]+"_opts.json","w")
    json.dump(opt,opt_file)
    opt_file.close()
    main(opt)
