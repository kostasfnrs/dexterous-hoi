from typing import Tuple
import numpy as np
import torch
from manopth.manolayer import ManoLayer
from sample.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle
from sample.visualize_joints import SkeletonVisualizer


def compute_hand_joints(
    hand_pose: np.ndarray, trans: np.ndarray, side: str = "right"
) -> np.ndarray:
    """
    Compute hand joints from MANO hand pose parameters.
    Args:
        hand_pose: np.ndarray of shape (B, 24)
        trans: np.ndarray of shape (B, 3)
        rot: np.ndarray of shape (3,)
    """
    mano_layer = ManoLayer(
        mano_root="body_models/mano",
        use_pca=True,
        ncomps=24,
        flat_hand_mean=True,
        side=side,
        center_idx=0,
    )
    hand_pose_tensor = torch.from_numpy(hand_pose)
    # add rotation to the end of the hand pose tensor
    # TODO: fix translation kwarg
    trans = np.asarray(trans)
    print(f"trans shape: {trans.shape}")
    print(f"hand_pose_tensor shape: {hand_pose_tensor.shape}")
    # _, hand_joints = mano_layer(hand_pose_tensor, th_trans=torch.from_numpy(trans))
    _, hand_joints = mano_layer(hand_pose_tensor)
    hand_joints = hand_joints / 1000.0
    hand_joints = hand_joints + torch.from_numpy(trans)[:, None, :]

    batch_size = hand_pose_tensor.shape[0]
    assert hand_joints.shape == (batch_size, 21, 3)

    return hand_joints.detach().numpy()


def convert_hand_to_axis_rot_hand(hand_pose: np.ndarray) -> np.ndarray:
    """
    Convert hand pose to Euler rotation.
    Args:
        hand_pose: np.ndarray of shape (N, 30)
    """
    # extract 6D rotation from hand pose
    rot = hand_pose[:, 24:]
    # convert to Euler angles
    rot_mat = rotation_6d_to_matrix(torch.tensor(rot))
    axis_angle_rot = matrix_to_axis_angle(rot_mat)
    # axis_angle_rot[:, [0, 1, 2]] = axis_angle_rot[:, [0, 2, 1]]

    # concatenate hand pose and axis angle rotation, rotation should be first three values
    hand_pose = np.concatenate(
        [axis_angle_rot.detach().numpy(), hand_pose[:, :24]], axis=1
    )

    return hand_pose


def generate_hand_joints(
    skeleton: np.ndarray, lhand_poses: np.ndarray, rhand_poses: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add PCA hands to the skeleton. Hand poses are 24 PCA pose parameters and 6D rotation.
    Args:
        skeleton: np.ndarray of shape (N, 22, 3)
        lhand_poses: np.ndarray of shape (N, 30)
        rhand_poses: np.ndarray of shape (N, 30)
    """
    lhand_axis_rot = convert_hand_to_axis_rot_hand(lhand_poses)
    rhand_axis_rot = convert_hand_to_axis_rot_hand(rhand_poses)

    lhand_wrists = skeleton[:, 20, :]
    rhand_wrists = skeleton[:, 21, :]
    lhand_joints = compute_hand_joints(lhand_axis_rot, lhand_wrists, side="left")
    rhand_joints = compute_hand_joints(rhand_axis_rot, rhand_wrists, side="right")

    return lhand_joints, rhand_joints, lhand_wrists, rhand_wrists
