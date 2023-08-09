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
from config import get_config
import json
from easydict import EasyDict as edict
from models.PointDSC import PointDSC
import numpy as np
import torch
from misc.fcgf import ResUNetBN2C as FCGF
from misc.cal_fcgf import extract_features
import open3d as o3d
import MinkowskiEngine as ME
from misc.knn import find_knn_gpu, nn_to_mutual, find_2nn
ROOT_DIR = Path(__file__).parent.parent

class PointDSCRegistration:
    def __init__(self):
        self.device = torch.device('cuda')
        self.pointdsc_model, self.pointdsc_config = self.load_pointdsc()
        self.config = self.pointdsc_config
        self.voxel_size = self.config.downsample
        self.fcgf_model = self.load_fcgf()

    def load_pointdsc(self):
        chosen_snapshot = "PointDSC_KITTI_release"
        config_path = os.path.join(ROOT_DIR, 'snapshot', chosen_snapshot, 'config.json')
        config = json.load(open(config_path, 'r'))
        config = edict(config)
        device = torch.device('cuda')
        model = PointDSC(
            in_dim=config.in_dim,
            num_layers=config.num_layers,
            num_channels=config.num_channels,
            num_iterations=config.num_iterations,
            ratio=config.ratio,
            sigma_d=config.sigma_d,
            k=config.k,
            nms_radius=config.inlier_threshold,
        ).to(device)
        miss = model.load_state_dict(
            torch.load(os.path.join(ROOT_DIR, 'snapshot', chosen_snapshot, 'models', 'model_best.pkl'), map_location=device), strict=False)
        print(miss)
        model.eval()
        return model, config

    def load_fcgf(self):
        weight_path = os.path.join(ROOT_DIR, 'misc', 'ResUNetBN2C-feat32-kitti-v0.3.pth')
        device = torch.device('cuda')
        fcgf_model = FCGF(
            1,
            32,
            bn_momentum=0.05,
            conv1_kernel_size=5,
            normalize_feature=True
        ).to(device)
        checkpoint = torch.load(weight_path)
        fcgf_model.load_state_dict(checkpoint['state_dict'])
        fcgf_model.eval()
        return fcgf_model

    def register(self, xyz0, xyz1):
        with torch.no_grad():
            # Step 0: voxelize and generate sparse input
            xyz0, coords0, feats0 = self.preprocess(xyz0)
            xyz1, coords1, feats1 = self.preprocess(xyz1)

            # Step 1: Feature extraction
            fcgf_feats0 = self.fcgf_feature_extraction(feats0, coords0)
            fcgf_feats1 = self.fcgf_feature_extraction(feats1, coords1)

            # Step 2: Coarse correspondences by mutual nearest neighbors
            corres_idx0, corres_idx1, idx1_2nd, _ = find_2nn(fcgf_feats0, fcgf_feats1)
            corres_idx0, corres_idx1, idx1_2nd = nn_to_mutual(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1, idx1_2nd, force_return_2nd=True)

            src_keypts = xyz0[corres_idx0, :]
            tgt_keypts = xyz1[corres_idx1, :]

            corr_pos = torch.cat([src_keypts, tgt_keypts], dim=-1)
            corr_pos = corr_pos - corr_pos.mean(0)

            # outlier rejection
            data = {
                'corr_pos': corr_pos.cuda().float().unsqueeze(0),
                'src_keypts': src_keypts.cuda().float().unsqueeze(0),
                'tgt_keypts': tgt_keypts.cuda().float().unsqueeze(0),
                'testing': True,
            }
            res = self.pointdsc_model(data)
            return res['final_trans'][0].detach().cpu().numpy()

    def fcgf_feature_matching(self, feats0, feats1):
        '''
        Step 2: coarsely match FCGF features to generate initial correspondences
        '''
        nns = find_knn_gpu(feats0,
                           feats1,
                           nn_max_n=self.network_config.nn_max_n,
                           knn=1,
                           return_distance=False)
        corres_idx0 = torch.arange(len(nns)).long().squeeze().to(self.device)
        corres_idx1 = nns.long().squeeze()

        return corres_idx0, corres_idx1

    def preprocess(self, pcd):
        '''
        Stage 0: preprocess raw input point cloud
        Input: raw point cloud
        Output: voxelized point cloud with
        - xyz:    unique point cloud with one point per voxel
        - coords: coords after voxelization
        - feats:  dummy feature placeholder for general sparse convolution
        '''
        if isinstance(pcd, o3d.geometry.PointCloud):
            xyz = np.array(pcd.points)
        elif isinstance(pcd, np.ndarray):
            xyz = pcd
        else:
            raise Exception('Unrecognized pcd type')

        # Voxelization:
        # Maintain double type for xyz to improve numerical accuracy in quantization
        _, sel = ME.utils.sparse_quantize(xyz / self.voxel_size, return_index=True)
        npts = len(sel)

        xyz = torch.from_numpy(xyz[sel]).to(self.device)

        # ME standard batch coordinates
        coords = ME.utils.batched_coordinates([torch.floor(xyz / self.voxel_size).int()], device=self.device)
        feats = torch.ones(npts, 1)

        return xyz.float(), coords, feats

    def fcgf_feature_extraction(self, feats, coords):
        '''
        Step 1: extract fast and accurate FCGF feature per point
        '''
        sinput = ME.SparseTensor(feats, coordinates=coords, device=self.device)

        return self.fcgf_model(sinput).F

    def fcgf_feature_matching(self, feats0, feats1):
        '''
        Step 2: coarsely match FCGF features to generate initial correspondences
        '''
        nns = find_knn_gpu(feats0,
                           feats1,
                           nn_max_n=250,
                           knn=1,
                           return_distance=False)
        corres_idx0 = torch.arange(len(nns)).long().squeeze().to(self.device)
        corres_idx1 = nns.long().squeeze()

        return corres_idx0, corres_idx1

def success(tf, gt):
    rot_est = tf[:3, :3]
    rot_gt = gt[:3, :3]
    trace = np.trace(np.dot(rot_est, rot_gt.T))
    tmp = np.clip((trace - 1) / 2, -1, 1)
    rot_succ = np.arccos(tmp) * 180 / np.pi < cfg.evaluation.rot_thd

    trans_est = tf[:3, 3]
    trans_gt = gt[:3, 3]
    trans_succ = np.linalg.norm(trans_gt-trans_est) < cfg.evaluation.trans_thd

    return rot_succ and trans_succ

def eval_test_file(test_file):

    pointdsc_reg = PointDSCRegistration()

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

    success_num = 0
    total_num = 0
    time_sum = 0
    results = {}
    kitti_dataset = KittiDataset(cfg.kitti_root)
    if "kitti360" in test_file:
        kitti_dataset = Kitti360Dataset(cfg.kitti360_root)
    for seq, i, j, tf_gt in tqdm(test_pairs, leave=False):
        source_cloud = kitti_dataset.get_lidar_pointcloud(seq, i)[:, :3]
        target_cloud = kitti_dataset.get_lidar_pointcloud(seq, j)[:, :3]
        t1 = perf_counter()
        # your registration code here
        tf = pointdsc_reg.register(source_cloud, target_cloud)
        t2 = perf_counter()

        # evaluate
        if not seq in results:
            results[seq] = {
                'success': 0,
                'total': 0,
                'time': 0,
            }
        results[seq]['success'] += success(tf, tf_gt)
        results[seq]['total'] += 1
        results[seq]['time'] += t2 - t1

        success_num += success(tf, tf_gt)
        total_num += 1
        time_sum += t2 - t1

    print("Test file: %s, total %d pairs" % (test_file, total_num))
    print("Success rate: %.2f%%, Average time: %.2fms" % (success_num / total_num * 100, time_sum / total_num * 1000))
    for seq in results:
        print("Seq %d: %.2f%%, Average time: %.2fms" % (seq, results[seq]['success'] / results[seq]['total'] * 100, results[seq]['time'] / results[seq]['total'] * 1000))



if __name__ == '__main__':

    args = get_config()

    cfg_from_yaml_file(os.path.join(cfg.ROOT_DIR, "configs/dataset.yaml"), cfg)

    test_file = os.path.join(cfg.ROOT_DIR, args.test_file)
    eval_test_file(test_file)


