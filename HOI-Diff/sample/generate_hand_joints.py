from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from manopth.manolayer import ManoLayer

### Pytorch3D functions
# Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#rotation_6d_to_matrix
# It is much more difficult to install PyTorch3D into the HOI-Diff env than to just copy the functions here.


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


### End Pytorch3D functions


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
    )
    hand_pose_tensor = torch.tensor(hand_pose, dtype=torch.float32).unsqueeze(0)
    # add rotation to the end of the hand pose tensor
    # TODO: fix translation kwarg
    hand_joints = mano_layer(hand_pose_tensor, trans=trans)
    return hand_joints.squeeze(0).detach().numpy()


def convert_hand_to_euler_rot(hand_pose: np.ndarray) -> np.ndarray:
    """
    Convert hand pose to Euler rotation.
    Args:
        hand_pose: np.ndarray of shape (N, 30)
    """
    # extract 6D rotation from hand pose
    rot = hand_pose[:, 24:]
    # convert to Euler angles
    rot_mat = rotation_6d_to_matrix(torch.tensor(rot))
    euler_rot = matrix_to_euler_angles(rot_mat, convention="XYZ")

    # concatenate hand pose and Euler rotation
    hand_pose = np.concatenate([hand_pose[:, :24], euler_rot.detach().numpy()], axis=1)

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
    lhand_euler = convert_hand_to_euler_rot(lhand_poses)
    rhand_euler = convert_hand_to_euler_rot(rhand_poses)

    print("skeleton shape")
    print(np.asarray(skeleton).shape)
    lhand_wrists = skeleton[:, 20, :]
    rhand_wrists = skeleton[:, 21, :]
    lhand_joints = compute_hand_joints(lhand_euler, lhand_wrists, side="left")
    rhand_joints = compute_hand_joints(rhand_euler, rhand_wrists, side="right")

    return lhand_joints, rhand_joints
