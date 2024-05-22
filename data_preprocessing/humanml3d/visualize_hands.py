import numpy as np
import matplotlib.pyplot as plt
import torch

from manopth.manolayer import ManoLayer
from visualize_skeleton import SkeletonVisualizer

def load_np_hand_data(file_path):
    return np.load(file_path, allow_pickle=True)

def compute_hand_joints_pca(pca_pose, global_orient, side='right'):
    mano_layer = ManoLayer(
        mano_root='body_models/mano', use_pca=True, ncomps=24, flat_hand_mean=True, side=side, center_idx=0
    )

    hand_pose_with_rot = torch.cat([global_orient, pca_pose], dim=-1)
    _, hand_joints = mano_layer(hand_pose_with_rot)
    hand_joints /= 1000.0

    return hand_joints.detach().cpu().numpy()

def visualize_hand_joints(hand_joints):
    visualizer = SkeletonVisualizer(hand_joints)
    visualizer.animate()
    

if __name__ == '__main__':
    # Load the hand data
    hand_data = load_np_hand_data('/media/erik/DATA/grab/grab_preprocessed/s10_airplane_fly_1/rhand_data.npz')
    # Compute the hand joints using PCA
    pca_poses = torch.from_numpy(hand_data['pca_pose']).float()
    global_orient = torch.from_numpy(hand_data['global_orient']).float().squeeze(0)

    print(pca_poses.shape, global_orient.shape)

    hand_joints = compute_hand_joints_pca(pca_poses, global_orient, side='right')
    # Visualize the hand joints
    visualize_hand_joints(hand_joints)

    