import json
import sys
import os
import numpy as np
import pandas as pd

video_label_name_path, logits_path, res_path, video_info, save_path = sys.argv[1:]
# video_label_name_path = './bsn/data/activitynet_annotations/action_name.csv'
# logits_path = 'output_ftrs_RGB_dpn92/v_00Dk03Jr70M.csv'
# res_path = './bsn/output/result_proposal.json'
# video_info = './video_info.csv'
# save_path = 'final_result.json'

action_name = pd.read_csv(video_label_name_path)
label2name = sorted(list(action_name.values.reshape(-1)))

with open(res_path) as f:
    results = json.load(f)
results = results['results']

video_info = pd.read_csv(video_info)
video_duration = video_info.seconds[0]
video_featureFrame = video_info.featureFrame[0]
video_name = video_info.video[0]+'.mp4'
result_info = {}
result_info['filename'] = video_name
result_info['duration'] = video_duration
new_res = []

logits = pd.read_csv(os.path.join(logits_path, video_info.video[0]+'.csv')).values

for key in results.keys():
    one_results = results[key]
    for result in one_results[:5]:
        start_idx = max(int(result['segment'][0]/video_duration*video_featureFrame/16), 0)
        end_idx = min(int(result['segment'][1]/video_duration*video_featureFrame/16), int(video_featureFrame/16))
        segment_logits = logits[start_idx:end_idx, :]
        label = np.argmax(np.bincount(np.argmax(segment_logits, axis=1)))
        result['label'] = label2name[label]
        new_res.append(result)
result_info['results'] = new_res
with open(save_path, 'w') as f:
    json.dump(result_info, f)
print 'Save result json in path: ', save_path