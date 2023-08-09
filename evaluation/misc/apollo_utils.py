import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from misc.config import cfg
from misc.utils import icp_registration, draw_pairs
from scipy.spatial.transform import Rotation as R

class ApolloDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.sessions = {
            20: "TestData/HighWay237/2018-10-12/",
            21: "TestData/SunnyvaleBigloop/2018-10-03/",
            22: "TestData/MathildaAVE/2018-10-12/",
            23: "TestData/SanJoseDowntown/2018-10-11/2/",
            24: "TestData/SanJoseDowntown/2018-10-11/1/",
            25: "TestData/BaylandsToSeafood/2018-10-12/",
            26: "TestData/ColumbiaPark/2018-10-11/",
        }
        self.session_poses = {}
        self.frame_ids = {}
        self.session_trajectories = {}
        for drive_id in self.sessions.keys():
            self.session_poses[drive_id] = {}
            pose_file = os.path.join(self.root_dir, self.sessions[drive_id], 'poses', 'gt_poses.txt')
            pose_array = np.genfromtxt(pose_file)
            self.frame_ids[drive_id] = pose_array[:, 0]
            for i in range(pose_array.shape[0]):
                tf = np.eye(4)
                tf[:3, :3] = R.from_quat(pose_array[i, 5:]).as_matrix()
                tf[:3, 3] = pose_array[i, 2:5]
                self.session_poses[drive_id][self.frame_ids[drive_id][i]] = tf
            self.session_trajectories[drive_id] = pose_array[:, 2:5]
        print("Apollo dataset loaded.")

    def get_lidar_pose(self, drive_id, frame_id):
        return self.session_poses[drive_id][frame_id]

    def read_transform(self, drive_id, idx_1, idx_2):
        # from idx_1 to idx_2
        pose1 = self.session_poses[drive_id][idx_1]
        pose2 = self.session_poses[drive_id][idx_2]
        return np.linalg.inv(pose2) @ pose1

    def refine_poses(self, pairs, icp_refine=False):
        refined_pairs = []
        for drive_id, i, j in tqdm(pairs):
            tf = self.read_transform(drive_id, i, j)
            if icp_refine:
                source_cloud = self.read_apollo_pc(drive_id, i)
                target_cloud = self.read_apollo_pc(drive_id, j)
                draw_pairs(source_cloud, target_cloud, tf, window_name="before")
                tf = icp_registration(source_cloud, target_cloud, tf)
                draw_pairs(source_cloud, target_cloud, tf, window_name="after")
            refined_pair = [drive_id, i, j]
            refined_pair.extend(tf.flatten())
            refined_pairs.append(refined_pair)
        return refined_pairs

    def read_apollo_pc(self, drive_id, id):
        pcd_path = os.path.join(cfg.apollo_root, self.sessions[drive_id], "pcds/%d.pcd" % id)
        points = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(points.points)
        return points