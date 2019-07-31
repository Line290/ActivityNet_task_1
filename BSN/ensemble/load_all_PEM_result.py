import os
import sys
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, Manager, Process

def load_all_PEM_results():
    folder_paths = os.listdir("/users/lindq/acti_comp/BSN-boundary-sensitive-network.pytorch/output/")
    folder_paths2 = os.listdir("/users/lindq/acti_comp/BSN-boundary-sensitive-network.pytorch/output_old/")
    folder_paths3 = os.listdir("/users/lindq/acti_comp/BSN-boundary-sensitive-network.pytorch/PEM_ensemble/")
    all_fold_pathes = []
    for fold_name in folder_paths:
        if "fold_2" in fold_name and "PEM" in fold_name:
            one_fold_path = os.path.join("/users/lindq/acti_comp/BSN-boundary-sensitive-network.pytorch/output/", fold_name)
            all_fold_pathes.append(one_fold_path)
    for fold_name in folder_paths2:
        one_fold_path = os.path.join("/users/lindq/acti_comp/BSN-boundary-sensitive-network.pytorch/output_old/", fold_name)
        num_items = os.listdir(one_fold_path)
        if len(num_items) == 9970:
            all_fold_pathes.append(one_fold_path)
    for fold_name in folder_paths3:
        one_fold_path = os.path.join("/users/lindq/acti_comp/BSN-boundary-sensitive-network.pytorch/PEM_ensemble/", fold_name)
        num_items = os.listdir(one_fold_path)
        if num_items == 9970:
            all_fold_pathes.append(one_fold_path)
    print("number of PEM results: ", len(all_fold_pathes))
    
    val_test_video_name = []
    for full_name in os.listdir(all_fold_pathes[0]):
        val_test_video_name.append(full_name[:-4])
    print("number of test video name: ", len(val_test_video_name))
    
    num_works = 64
    #init dict
    global all_PEM_results
    all_PEM_results = Manager().dict()
#     for name in val_test_video_name:
#         all_PEM_results[name] = []
    def save_all_results_for_one_name(name_list, all_fold_pathes):
        for name in name_list:
            df_list = []
            for one_fold_path in all_fold_pathes:
                tmp_df = pd.read_csv(os.path.join(one_fold_path, name+'.csv'))
        #         print(tmp_df)
                df_list.append(tmp_df)
            all_PEM_results[name] = df_list
    P = []
    for i in range(num_works+1):
        start_idx = int(len(val_test_video_name) / num_works * i)
        end_idx = min(int(len(val_test_video_name) / num_works * (i+1)), len(val_test_video_name))
        p = Process(target=save_all_results_for_one_name, args=(val_test_video_name[start_idx:end_idx], all_fold_pathes))
        p.start()
        P.append(p)
    for p in P:
        p.join()
    return all_PEM_results, len(all_fold_pathes), all_fold_pathes
if __name__ == '__main__':
    a, b, c = load_all_PEM_results()
    print(len(dict(a)['v__3I4nm2zF5Y']))
