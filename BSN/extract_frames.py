import os
import argparse
import cv2
import json
import math
from tqdm import tqdm
from skvideo.io import vread


def file_path(path):
    return os.path.abspath(os.path.expanduser(path))


def extract(video_path, positions):
    if not isinstance(positions, (list, tuple)):
        raise RuntimeError('expect list or tuple, get {}.'.format(type(positions)))

    capture = cv2.VideoCapture(video_path)
    frames = []
    for pos in positions:
        total_len = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
        if pos >= total_len:
            continue

        capture.set(cv2.CAP_PROP_POS_MSEC, pos * 1000)
        ok, img = capture.read()
        if not ok:
            img = vread(video_path)[math.floor(capture.get(cv2.CAP_PROP_FPS) * pos)]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # raise RuntimeError(video_path, pos)
        if img.shape[0] > img.shape[1] > 224:
            img = cv2.resize(img, (224, round(img.shape[0] / img.shape[1] * 224)), interpolation=cv2.INTER_CUBIC)
        elif img.shape[1] > img.shape[0] > 224:
            img = cv2.resize(img, (round(img.shape[1] / img.shape[0] * 224), 224), interpolation=cv2.INTER_CUBIC)
        frames.append(img)
    capture.release()
    return frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', '-v', type=file_path, required=True)
    parser.add_argument('--json_file', '-json', type=file_path, required=True)
    parser.add_argument('--output_dir', '-o', type=file_path)
    parser.add_argument('--num_frames', '-n', type=int, default=3)
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = os.path.join(os.path.dirname(args.video_dir), 'frames')

    with open(args.json_file, 'r') as f:
        database = json.load(f)['database']

    for vid, item in tqdm(database.items()):
        subset = item['subset']
        if not subset in ['training', 'validation']:
            continue

        for anno in item['annotations']:
            video_path = os.path.join(args.video_dir, 'v_{}.mp4'.format(vid))
            label = anno['label']
            segment = anno['segment']

            seg_len = segment[1] - segment[0]
            if seg_len > 0.01 and os.path.exists(video_path):
                positions = [segment[0] + (i + 1) * seg_len / (args.num_frames + 1) for i in range(args.num_frames)]
                frames = extract(video_path, positions)

                os.makedirs(os.path.join(args.output_dir, subset, label), exist_ok=True)
                for img in frames:
                    idx = 0
                    img_path = os.path.join(args.output_dir, subset, label, '{}_{}.jpg'.format(vid, idx))
                    while os.path.exists(img_path):
                        idx += 1
                        img_path = os.path.join(args.output_dir, subset, label, '{}_{}.jpg'.format(vid, idx))
                    cv2.imwrite(img_path, img)
