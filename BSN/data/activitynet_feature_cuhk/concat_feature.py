import os
import argparse
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool


def concat(csv_name):
    paths = [os.path.join(dpath, csv_name) for dpath in args.input_dirs]
    dfs = [pd.read_csv(path).ix[:, :200] for path in paths]
    seg = [0]
    for df in dfs:
#         seg.append(seg[-1] + df.shape[1])
        seg.append(seg[-1] + 200)

    for i, df in enumerate(dfs):
        df.columns = ['f{}'.format(j) for j in range(seg[i], seg[i+1])]

    df_all = pd.concat(dfs, axis=1)
    df_all.to_csv(os.path.join(args.output_dir, csv_name), index=False)


def file_path(path):
    return os.path.abspath(os.path.expanduser(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', '-i', nargs='+', required=True)
    parser.add_argument('--output_dir', '-o', required=True)
    parser.add_argument('--num_workers', '-j', type=int, default=64)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        
    csv_list = [fname for fname in os.listdir(args.input_dirs[0]) if fname.startswith('v_') and fname.endswith('.csv')]
    tqdm.write('total: {}'.format(len(csv_list)))

    pool = Pool(args.num_workers)
    with tqdm(total=len(csv_list)) as pbar:
        for _ in pool.imap_unordered(concat, csv_list):
            pbar.update()
