from typing import Tuple
import numpy as np
import torch

from manopth.manolayer import ManoLayer
from sample.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle
from sample.visualize_joints import SkeletonVisualizer
# import mano


def compute_hand_joints(
    hand_pose: np.ndarray,
    global_orient: np.ndarray,
    trans: np.ndarray,
    side: str = "right",
) -> np.ndarray:
    """
    Compute hand joints from MANO hand pose parameters.
    Args:
        hand_pose: np.ndarray of shape (B, 24)
        global_orient: np.ndarray of shape (B, 3)
        trans: np.ndarray of shape (B, 3)
    """
    mano_layer = ManoLayer(
        mano_root="body_models/mano",
        use_pca=True,
        ncomps=24,
        flat_hand_mean=True,
        side=side,
        center_idx=0,
    )
    # mano_model = mano.load(
    #     model_path="body_models/mano",
    #     is_rhand=(side == "right"),
    #     num_pca_comps=24,
    #     flat_hand_mean=False,
    #     batch_size=hand_pose.shape[0],
    # output = mano_model(
    #     hand_pose=th_hand_pose,
    #     global_orient=th_global_orient,
    #     transl=th_trans,
    #     return_tips=True,
    #     return_verts=True,
    # )  # )

    th_hand_pose = torch.from_numpy(hand_pose)
    th_global_orient = torch.from_numpy(global_orient)
    th_trans = torch.from_numpy(trans)

    hand_pose_with_rot = torch.cat([th_global_orient, th_hand_pose], dim=1)
    _, hand_joints = mano_layer(hand_pose_with_rot)
    hand_joints = hand_joints / 1000.0
    hand_joints = hand_joints + th_trans[:, None, :]

    batch_size = hand_pose.shape[0]
    assert hand_joints.shape == (batch_size, 21, 3)

    return hand_joints.detach().numpy()


def extract_global_orient(
    hand_pose: np.ndarray, rot_mode: str = "cont_6d"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert hand pose to Euler rotation.
    Args:
        hand_pose: np.ndarray of shape (N, 30)
    """
    if rot_mode == "cont_6d":
        # extract 6D rotation from hand pose
        rot = hand_pose[:, :6]
        # convert to Euler angles
        rot_mat = rotation_6d_to_matrix(torch.tensor(rot))
        # rot_mat[:, [0, 1, 2]] = rot_mat[:, [0, 2, 1]]

        # switch y and z axes with permutation matrix
        permute = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.float32)
        rot_mat = torch.matmul(rot_mat, permute)
        axis_angle_rot = matrix_to_axis_angle(rot_mat)
        axis_angle_rot = axis_angle_rot.detach().numpy()
        hand_pose_vec = hand_pose[:, 6:]
    elif rot_mode == "axisang":
        axis_angle_rot = hand_pose[:, :3]
        hand_pose_vec = hand_pose[:, 3:]
    else:
        raise NotImplementedError(f"Invalid rot_mode: {rot_mode}")

    return axis_angle_rot, hand_pose_vec


def generate_hand_joints(
    skeleton: np.ndarray,
    lhand_poses: np.ndarray,
    rhand_poses: np.ndarray,
    hand_mode: str = "joints",
    rot_mode: str = "cont_6d",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Add PCA hands to the skeleton. Hand poses are 24 PCA pose parameters and 6D rotation.
    Args:
        skeleton: np.ndarray of shape (N, 22, 3)

        If hand_mode is 'joints':
        Hand poses are flattened 21x3 hand joint positions, centered around the wrist
        lhand_poses: np.ndarray of shape (N, 63)
        rhand_poses: np.ndarray of shape (N, 63)

        If hand_mode is 'PCA':
        Hand poses are [6d rotation, 24d hand pose]
        lhand_poses: np.ndarray of shape (N, 30)
        rhand_poses: np.ndarray of shape (N, 30)
    """
    if hand_mode == "PCA":
        lhand_global_orient, lhand_poses = extract_global_orient(lhand_poses, rot_mode)
        rhand_global_orient, rhand_poses = extract_global_orient(rhand_poses, rot_mode)

        lhand_wrists = skeleton[:, 20, :]
        rhand_wrists = skeleton[:, 21, :]
        lhand_joints = compute_hand_joints(
            lhand_poses, lhand_global_orient, lhand_wrists, side="left"
        )
        rhand_joints = compute_hand_joints(
            rhand_poses, rhand_global_orient, rhand_wrists, side="right"
        )
    elif hand_mode == "joints":
        lhand_wrists = skeleton[:, 20, :]
        rhand_wrists = skeleton[:, 21, :]
        lhand_poses = torch.from_numpy(lhand_poses)
        rhand_poses = torch.from_numpy(rhand_poses)

        lhand_joints = lhand_poses.unflatten(1, (21, 3)) + lhand_wrists[:, None, :]
        rhand_joints = rhand_poses.unflatten(1, (21, 3)) + rhand_wrists[:, None, :]

        # from sample.visualize_joints import SkeletonVisualizer
        #
        # normalized_rhand_joints = torch.zeros_like(rhand_joints)
        # for i, val in enumerate(rhand_joints):
        #     normalized_rhand_joints[i] = normalize_points(val)
        #
        # normalized_rhand_joints = np.array(normalized_rhand_joints)
        # vis = SkeletonVisualizer(normalized_rhand_joints)
        # vis.animate()
        #
    else:
        raise NotImplementedError(f"Invalid hand_mode: {hand_mode}")

    return lhand_joints, rhand_joints, lhand_wrists, rhand_wrists
