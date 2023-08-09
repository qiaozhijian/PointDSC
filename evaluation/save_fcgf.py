import numpy as np
import sys
import os
os.environ["OMP_NUM_THREADS"] = "16"
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from misc.config import cfg_from_yaml_file, cfg
sys.path.append(cfg.ROOT_DIR)
from misc.kitti_utils import KittiDataset
from misc.kitti360_utils import Kitti360Dataset
from tqdm import tqdm
from time import perf_counter
import argparse
from core import load_dgr
from config import get_config

def eval_test_file(test_file):

    # read test file
    test_pairs = []
    for row, line in enumerate(open(test_file)):
        if row == 0:
            continue
        line = line.strip()
        if len(line) == 0:
            continue
        line = line.split()
        seq, i, j = int(line[0]), int(line[1]), int(line[2])
        tf = np.array([float(x) for x in line[3:19]]).reshape(4, 4)
        test_pairs.append((seq, i, j, tf))

    kitti_dataset = KittiDataset(cfg.kitti_root)
    if "kitti360" in test_file:
        kitti_dataset = Kitti360Dataset(cfg.kitti360_root)
    dgr = load_dgr()
    total_num = 0
    time_sum = 0
    for seq, i, j, tf_gt in tqdm(test_pairs, leave=False):
        source_cloud = kitti_dataset.get_lidar_pointcloud(seq, i)[:, :3]
        target_cloud = kitti_dataset.get_lidar_pointcloud(seq, j)[:, :3]
        t1 = perf_counter()
        # your registration code here
        correspondeces = dgr.get_correspondences(source_cloud, target_cloud)
        t2 = perf_counter()

        if "kitti360" in test_file:
            save_dir = os.path.join(kitti_dataset.velo_dir, "2013_05_28_drive_{:04d}_sync".format(seq), "velodyne_points", "correspondences")
        else:
            save_dir = os.path.join(kitti_dataset.kitti_root, 'sequences/%02d/correspondences' % seq)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{i}_{j}.txt')
        np.savetxt(save_path, correspondeces.astype(np.float32), fmt='%.3f')
        total_num += 1
        time_sum += t2 - t1
    print("Average time: %.3f" % (time_sum / total_num))

if __name__ == '__main__':

    args = get_config()

    cfg_from_yaml_file(os.path.join(cfg.ROOT_DIR, "configs/dataset.yaml"), cfg)

    test_file = os.path.join(cfg.ROOT_DIR, args.test_file)
    eval_test_file(test_file=test_file)


