import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from misc.config import cfg
from misc.utils import icp_registration, draw_pairs, Rodrigues, curl_velodyne_data
from scipy.spatial.transform import Rotation as R

import glob
class Kitti360Dataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.velo_dir = os.path.join(root_dir, "data_3d_raw")
        self.pose_dir = os.path.join(root_dir, "data_poses")
        self.lidar_to_pose = self.get_lidar_to_imu()
        self.session_poses = {}
        self.session_trajecotries = {}
        self.session_frame_ids = {}
        for sessions in glob.glob(os.path.join(self.pose_dir, "*")):
            seq = int(sessions.split('/')[-1].split('_')[4])
            self.session_poses[seq] = {}
        for sessions in glob.glob(os.path.join(self.pose_dir, "*")):
            seq = int(sessions.split('/')[-1].split('_')[4])
            pose_file = os.path.join(sessions, "poses.txt")
            pose_array = np.genfromtxt(pose_file).reshape((-1, 13))
            self.session_trajecotries[seq] = pose_array[:, [4, 8, 12]]
            self.session_frame_ids[seq] = pose_array[:, 0].astype(np.int32)
            for i in range(len(pose_array)):
                frame_id = int(pose_array[i, 0])
                pose = np.eye(4)
                pose[:3, :4] = pose_array[i, 1:].reshape((3, 4))
                self.session_poses[seq][frame_id] = pose @ self.lidar_to_pose
        print("Finish loading poses")

    def get_lidar_pose(self, seq, frame_id):
        return self.session_poses[seq][frame_id]

    def check_pc_exist(self, seq, frame_id):
        velo_file = os.path.join(self.velo_dir, "2013_05_28_drive_{:04d}_sync".format(seq), "velodyne_points", "data", "{:010d}.bin".format(frame_id))
        return os.path.exists(velo_file)

    def get_lidar_pointcloud(self, seq, frame_id):
        velo_file = os.path.join(self.velo_dir, "2013_05_28_drive_{:04d}_sync".format(seq), "velodyne_points", "data", "{:010d}.bin".format(frame_id))
        velo = np.fromfile(velo_file, dtype=np.float32).reshape((-1, 4))
        ego_motion = self.get_ego_motion(seq, frame_id)
        if np.linalg.norm(ego_motion - np.eye(4)) > 0.1:
            r = Rodrigues(ego_motion[0:3, 0:3])
            t = ego_motion[0:3, 3]
            velo = curl_velodyne_data(velo, r, t)

        return velo

    def get_ego_motion(self, seq, frame_id):
        if frame_id in self.session_poses[seq] and frame_id - 1 in self.session_poses[seq]:
            return np.linalg.inv(self.session_poses[seq][frame_id]) @ self.session_poses[seq][frame_id - 1]
        elif frame_id in self.session_poses[seq] and frame_id + 1 in self.session_poses[seq]:
            return np.linalg.inv(self.session_poses[seq][frame_id + 1]) @ self.session_poses[seq][frame_id]
        else:
            return np.eye(4)
    
    def get_lidar_to_imu(self):
        calib_cam_to_pose_file = os.path.join(self.root_dir, "calibration", "calib_cam_to_pose.txt")
        calib_cam_to_velo_file = os.path.join(self.root_dir, "calibration", "calib_cam_to_velo.txt")
        calib_cam_to_pose = np.eye(4)
        with open(calib_cam_to_pose_file, 'r') as f:
            line = f.readline()
            line = line[9:].strip().split(' ')
            calib_cam_to_pose[:3, :] = np.array(line).astype(np.float32).reshape((3, 4))
        calib_cam_to_velo = np.eye(4)
        with open(calib_cam_to_velo_file, 'r') as f:
            line = f.readline()
            line = line.strip().split(' ')
            calib_cam_to_velo[:3, :] = np.array(line).astype(np.float32).reshape((3, 4))
        calib_lidar_to_pose = calib_cam_to_pose @ np.linalg.inv(calib_cam_to_velo)
        return calib_lidar_to_pose

    def refine_poses(self, pairs, icp_refine=False):
        refined_pairs = []
        for idx, p in enumerate(tqdm(pairs)):
            seq, frame_id_i, frame_id_j = p
            if not self.check_pc_exist(seq, frame_id_i) or not self.check_pc_exist(seq, frame_id_j):
                continue
            tf = np.linalg.inv(self.get_lidar_pose(seq, frame_id_j)) @ self.get_lidar_pose(seq, frame_id_i)
            if icp_refine:
                source_cloud = self.get_lidar_pointcloud(seq, frame_id_i)
                target_cloud = self.get_lidar_pointcloud(seq, frame_id_j)
                # draw_pairs(source_cloud, target_cloud, tf, window_name="before")
                tf = icp_registration(source_cloud, target_cloud, tf)
                # draw_pairs(source_cloud, target_cloud, tf, window_name="after")
            refined_pair = [seq, frame_id_i, frame_id_j]
            refined_pair.extend(tf.flatten())
            refined_pairs.append(refined_pair)
        return refined_pairs

    def read_trajectory(self, session):
        pose_file = os.path.join(self.root_dir, "data_poses", session, 'poses.txt')
        pose_array = np.genfromtxt(pose_file).reshape((-1, 13))
        frame_ids = pose_array[:, 0].astype(np.int32)
        pose_array = pose_array[:, 1:].reshape((-1, 3, 4))
        trajectory = []
        lidar_to_pose = self.get_lidar_to_imu(self.root_dir)
        for i in range(pose_array.shape[0]):
            tf = np.eye(4)
            tf[:3, :3] = pose_array[i, :3, :3]
            tf[:3, 3] = pose_array[i, :3, 3]
            tf = tf @ lidar_to_pose
            trajectory.append(tf)
        trajectory = np.array(trajectory)
        trajectory = trajectory[:, :3, 3]
        return trajectory, frame_ids