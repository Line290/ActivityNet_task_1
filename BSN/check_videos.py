import os
import json
import argparse
import cv2
from tqdm import tqdm


if __name__ == '__main__':
    file_path = lambda path: os.path.abspath(os.path.expanduser(path))

    parser = argparse.ArgumentParser()
    parser.add_argument('video_dir', type=file_path)
    parser.add_argument('json_file', type=file_path)
    args = parser.parse_args()

    with open(args.json_file) as f:
        database = json.load(f)['database']

    for vid, annotations in tqdm(database.items()):
        fname = 'v_{}.mp4'.format(vid)
        duration = annotations['duration']

        capture = cv2.VideoCapture(os.path.join(args.video_dir, fname))
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)

        actual_duration = count / fps
        if duration - actual_duration > 3:
            tqdm.write('{}: actual_duration={:.2f}, duration={}'.format(fname, actual_duration, duration))

        capture.release()

