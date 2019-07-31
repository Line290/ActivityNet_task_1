import os
import sys
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

rgb_folder, flow_folder, fuse_folder, num_works = sys.argv[1:]
num_works = int(num_works)
if os.path.exists(fuse_folder) is False:
    os.mkdir(fuse_folder)
name_list = os.listdir(rgb_folder)

def fuse_rgb_and_flow(video_list):
    for name in tqdm(video_list):
        one_fuse_path = os.path.join(fuse_folder, name)
        if os.path.exists(one_fuse_path):
            continue
        one_rgb_path = os.path.join(rgb_folder, name)
        one_flow_path = os.path.join(flow_folder, name)
        rgb_feat = pd.read_csv(one_rgb_path)
        flow_feat = pd.read_csv(one_flow_path)
        fuse = pd.concat([rgb_feat, flow_feat], axis=1)
#         one_fuse_path = os.path.join(fuse_folder, name)
        fuse.to_csv(one_fuse_path, index=None)
    
num_names_per_thread = len(name_list) / num_works
processes = []

for tid in range(int(num_works)):
    start = tid*num_names_per_thread
    end = min((tid+1)*num_names_per_thread, len(name_list))
    tmp_name_list = name_list[start: end]
    p = mp.Process(target=fuse_rgb_and_flow, args=(tmp_name_list,))
    p.start()
    processes.append(p)
for p in processes:
    p.join()
