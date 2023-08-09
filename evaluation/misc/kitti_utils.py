import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from misc.config import cfg
from misc.utils import icp_registration

class KittiDataset:
    def __init__(self, kitti_root, poses_root=None):
        self.kitti_root = kitti_root
        if poses_root is None:
            poses_root = os.path.join(kitti_root, "poses")
        self.poses_root = poses_root
        self.seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.lidar_to_cam = {}
        for seq in self.seqs:
            self.lidar_to_cam[seq] = self.get_lidar_to_cam(seq)
        self.seq_poses = {}
        self.session_trajecotries = {}
        for seq in self.seqs:
            pose_file = os.path.join(self.poses_root, "%02d.txt" % seq)
            pose_array = np.genfromtxt(pose_file).reshape((-1, 3, 4))
            self.session_trajecotries[seq] = pose_array[:, :3, 3]
            self.seq_poses[seq] = {}
            for i in range(pose_array.shape[0]):
                tf = np.eye(4)
                tf[:3, :3] = pose_array[i][:3, :3]
                tf[:3, 3] = pose_array[i][:3, 3]
                self.seq_poses[seq][i] = tf @ self.lidar_to_cam[seq]
        print("Kitti dataset loaded.")

    def get_lidar_pose(self, seq, frame_id):
        return self.seq_poses[seq][frame_id]

    def get_lidar_pc(self, seq, frame_id):
        pc_file = os.path.join(self.kitti_root, "sequences/%02d/velodyne/%06d.bin" % (seq, frame_id))
        pc = np.fromfile(pc_file, dtype=np.float32).reshape((-1, 4))
        return pc

    def get_lidar_to_cam(self, seq):
        calib_file = os.path.join(self.kitti_root, "sequences/%02d" % seq, "calib.txt")
        with open(calib_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith("Tr:"):
                    line = line.split(" ")[1:]
                    tf = np.array(line, dtype=np.float32).reshape((3, 4))
                    tf = np.vstack((tf, np.array([0, 0, 0, 1], dtype=np.float32)))
                    return tf

    def get_lidar_pointcloud(self, seq, id):
        bin_path = os.path.join(self.kitti_root, "sequences/%02d" % seq, "velodyne/%06d.bin" % id)
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        return points

    def get_lidar_pose(self, seq, idx):
        return self.seq_poses[seq][idx]

    def get_tf(self, seq, idx_1, idx_2):
        # from idx_1 to idx_2
        pose1 = self.get_lidar_pose(seq, idx_1)
        pose2 = self.get_lidar_pose(seq, idx_2)
        tf = np.eye(4)
        tf[:3, :3] = pose2[:3, :3].T.dot(pose1[:3, :3])
        tf[:3, 3] = pose2[:3, :3].T @ (pose1[:3, 3] - pose2[:3, 3])
        return tf

    def refine_poses(self, pairs, refine = True):
        refined_pairs = []
        for seq, i, j in tqdm(pairs):
            tf = self.get_tf(seq, i, j)
            # draw_pairs(source_cloud, target_cloud, tf, window_name="before")
            if refine:
                source_cloud = self.read_kitti_pc(seq, i)
                target_cloud = self.read_kitti_pc(seq, j)
                tf = icp_registration(source_cloud, target_cloud, tf)
            # draw_pairs(source_cloud, target_cloud, tf, window_name="after")
            refined_pair = [seq, i, j]
            refined_pair.extend(tf.flatten())
            refined_pairs.append(refined_pair)
        return refined_pairs

def read_trajectory(poses_path, seq):
    pose_file = os.path.join(poses_path, "%02d.txt" % seq)
    pose_array = np.genfromtxt(pose_file)
    pose_array = pose_array.reshape((-1, 3, 4))
    trajectory = []
    for i in range(pose_array.shape[0]):
        tf = np.eye(4)
        tf[:3, :3] = pose_array[i][:3, :3]
        tf[:3, 3] = pose_array[i][:3, 3]
        trajectory.append(tf)
    trajectory = np.array(trajectory)
    trajectory = trajectory[:, :3, 3]
    return trajectory