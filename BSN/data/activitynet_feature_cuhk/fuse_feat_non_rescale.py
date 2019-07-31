import os
import sys
import pandas as pd
from tqdm import tqdm
rgb_folder, flow_folder, fuse_folder = sys.argv[1:]
if os.path.exists(fuse_folder) is False:
    os.mkdir(fuse_folder)
name_list = os.listdir(rgb_folder)
for name in tqdm(name_list):
    one_fuse_path = os.path.join(fuse_folder, name)
    if os.path.exists(one_fuse_path):
        continue
    one_rgb_path = os.path.join(rgb_folder, name)
    one_flow_path = os.path.join(flow_folder, name)
    rgb_feat = pd.read_csv(one_rgb_path)
    flow_feat = pd.read_csv(one_flow_path)
    fuse = pd.concat([rgb_feat, flow_feat], axis=1)
    #one_fuse_path = os.path.join(fuse_folder, name)
    fuse.to_csv(one_fuse_path, index=None)
