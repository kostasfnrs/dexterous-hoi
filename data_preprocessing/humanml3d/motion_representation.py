from os.path import join as pjoin

from common.skeleton import Skeleton
import numpy as np
import os
from common.quaternion import *
from paramUtil import *
import codecs as cs
import torch
from tqdm import tqdm
import os
from trimesh import Trimesh
import trimesh
from scipy.spatial.transform import Rotation

# from skel_vis import plot
from scipy import signal

simplified_mesh = {
    "backpack": "backpack/backpack_f1000.ply",
    "basketball": "basketball/basketball_f1000.ply",
    "boxlarge": "boxlarge/boxlarge_f1000.ply",
    "boxtiny": "boxtiny/boxtiny_f1000.ply",
    "boxlong": "boxlong/boxlong_f1000.ply",
    "boxsmall": "boxsmall/boxsmall_f1000.ply",
    "boxmedium": "boxmedium/boxmedium_f1000.ply",
    "chairblack": "chairblack/chairblack_f2500.ply",
    "chairwood": "chairwood/chairwood_f2500.ply",
    "monitor": "monitor/monitor_closed_f1000.ply",
    "keyboard": "keyboard/keyboard_f1000.ply",
    "plasticcontainer": "plasticcontainer/plasticcontainer_f1000.ply",
    "stool": "stool/stool_f1000.ply",
    "tablesquare": "tablesquare/tablesquare_f2000.ply",
    "toolbox": "toolbox/toolbox_f1000.ply",
    "suitcase": "suitcase/suitcase_f1000.ply",
    "tablesmall": "tablesmall/tablesmall_f1000.ply",
    "yogamat": "yogamat/yogamat_f1000.ply",
    "yogaball": "yogaball/yogaball_f1000.ply",
    "trashbin": "trashbin/trashbin_f1000.ply",
}


def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    # print(src_offset)
    # print(tgt_offset)
    """Calculate Scale Ratio as the ratio of legs"""
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    """Inverse Kinematics"""
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    """Forward Kinematics"""
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints, scale_rt


def process_file(positions, positions_obj, feet_thre):
    """
    root information (first 3 values) is absolute instead of relative.
    """
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    # '''Uniform Skeleton'''
    # positions, scale_rt = uniform_skeleton(positions, tgt_offsets)

    # # mirror z and y axis
    # positions[:, :, 2] *= -1
    # positions[:, :, 1] *= -1
    # # also mirror the rotations of the object
    # # positions_obj[:, [2, 4]] *= -1
    # # positions_obj[:, [1, 3]] *= -1

    # # swap y and z
    # positions = positions[:, :, [0, 2, 1]]
    # positions_obj = positions_obj[:, [0, 2, 1, 3, 5, 1]]

    # # swap x and z
    # positions = positions[:, :, [2, 1, 0]]
    # positions_obj = positions_obj[:, [2, 1, 0, 5, 4, 3]]

    # swap

    """Put on Floor"""
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    # --------handle obj rot + trans and hands
    positions_obj[:, 4] -= floor_height

    # plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

    """XZ at origin"""
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz
    # # --------handle obj rot + trans
    positions_obj[:, 3:] = positions_obj[:, 3:] - root_pose_init_xz

    # # '''Move the first pose to origin '''
    # # root_pos_init = positions[0]
    # # positions = positions - root_pos_init[0]

    # '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = (
        forward_init / np.sqrt((forward_init**2).sum(axis=-1))[..., np.newaxis]
    )

    #     print(forward_init)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init_for_all = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()

    positions = qrot_np(root_quat_init_for_all, positions)
    # --------handle obj rot + trans
    positions_obj[:, 3:] = qrot_np(root_quat_init_for_all[:, 0], positions_obj[:, 3:])

    # plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)

    """New ground truth positions"""
    global_positions = positions.copy()

    #     plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    #     plt.plot(positions[:, 0, 0], positions[:, 0, 2], marker='o', color='r')
    #     plt.xlabel('x')
    #     plt.ylabel('z')
    #     plt.axis('equal')
    #     plt.show()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([0.05, 0.025])
        # velfactor, heightfactor = np.array(
        # [thres, thres]), np.array([3.0, 2.0])
        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1, fid_l, 1]
        feet_l = (
            ((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)
        ).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1, fid_r, 1]
        feet_r = (
            ((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)
        ).astype(np.float32)
        return feet_l, feet_r

    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    """Quaternion and Cartesian representation"""
    r_rot = None

    def get_rifke(positions):
        """Local pose"""
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        """All pose face Z+"""
        positions = qrot_np(
            np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions
        )
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(
            positions, face_joint_indx, smooth_forward=False
        )

        """Fix Quaternion Discontinuity"""
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        """Root Linear Velocity"""
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        """Root Angular Velocity"""
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(
            positions, face_joint_indx, smooth_forward=True
        )

        """Quaternion to continuous 6D"""
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        """Root Linear Velocity"""
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        """Root Angular Velocity"""
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    """Root height"""
    root_y = positions[:, 0, 1:2]

    """Root rotation and linear velocity"""
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    """Get Joint Rotation Representation"""
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    """Get Joint Rotation Invariant Position Represention"""
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    """Get Joint Velocity Representation"""
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(
        np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
        global_positions[1:] - global_positions[:-1],
    )
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(data.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity, positions_obj[:-1]


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data, return_rot_ang=False):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    """Get Y-axis rotation from rotation velocity"""
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    """Add Y-axis rotation to root position"""
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]

    if return_rot_ang:
        return r_rot_quat, r_pos, r_rot_ang
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4 : (joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    """Add Y-axis rotation to local joints"""
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions
    )

    """Add root XZ to joints"""
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    """Concate root and joints"""
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


"""
For HumanML3D Dataset
"""

if __name__ == "__main__":
    # example_id = "Date03_Sub04_chairblack_liftreal_3"
    example_id = "Date07_Sub08_yogamat"
    # Lower legs
    l_idx1, l_idx2 = 5, 8
    # Right/Left foot
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    # l_hip, r_hip
    r_hip, l_hip = 2, 1
    joints_num = 22
    # ds_num = 8

    data_root = "/media/erik/DATA"
    # stage_1_output = f'{data_root}/behave_processed/pose_data_behave'
    # stage_2_output = f'{data_root}/behave_processed/joints_behave'
    # save_dir1 = f'{data_root}/behave_processed/new_joints_local'
    # save_dir2 = f'{data_root}/behave_processed/new_joint_vecs_local'

    # stage_1_output = './data/pose_data_behave'
    # stage_2_output = './data/joints_behave'
    # save_dir1 = './data/new_joints_local/'
    # save_dir2 = './data/new_joint_vecs_local/'

    stage_1_output = f"{data_root}/grab/pose_data_grab"
    stage_2_output = f"{data_root}/grab/joints_grab"
    save_dir1 = f"{data_root}/grab/new_joints_local"
    save_dir2 = f"{data_root}/grab/new_joint_vecs_local"
    data_dir = stage_2_output
    data_obj_dir = stage_1_output

    # save_dir1 = '/media/erik/DATA/new_joints_local/'
    # save_dir2 = '/media/erik/DATA/new_joint_vecs_local/'

    os.makedirs(save_dir1, exist_ok=True)
    os.makedirs(save_dir2, exist_ok=True)

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain

    # # Get offsets of target skeleton
    # example_data = np.load(os.path.join(data_dir, example_id + '.npy'))
    # example_data = example_data.reshape(len(example_data), -1, 3)
    # example_data = torch.from_numpy(example_data)
    # tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    # # (joints_num, 3)
    # tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
    # print(tgt_offsets)

    source_list = os.listdir(data_dir)
    frame_num = 0
    for source_file in tqdm(source_list):
        root_data = np.load(os.path.join(data_dir, source_file))

        source_data = root_data[:, :joints_num]

        lhand_file = os.path.join(data_obj_dir, source_file[:-4], "lhand_data.npy")
        rhand_file = os.path.join(data_obj_dir, source_file[:-4], "rhand_data.npy")

        source_data_lhand = np.load(lhand_file)
        source_data_rhand = np.load(rhand_file)

        source_data_obj = np.load(
            os.path.join(data_obj_dir, source_file[:-4], "object_fit_all.npy")
        )
        source_data = source_data.reshape(len(source_data), -1, 3)
        # try:
        # compute absoluate root information instead of relative, ignore rec_ric_data
        data, ground_positions, positions, l_velocity, data_obj = process_file(
            source_data, source_data_obj, 0.002
        )

        rec_ric_data = recover_from_ric(
            torch.from_numpy(data).unsqueeze(0).float(), joints_num
        )
        seq_len = data.shape[0]
        # 263 human + 6 obj + 24 left hand + 24 right hand
        # Date06_Sub07_backpack_back.npy he array at index 0 has size 1440 and the array at index 1 has size 1411

        # the data is cropped by one because of the velocity computation, the same cropped is applied to the object data in process_file
        data = np.concatenate([data, data_obj, source_data_lhand[:-1], source_data_rhand[:-1]], axis=-1)

        # # load obj points----------------
        # obj_name = source_file.split('_')[2]
        # obj_path = '/work/vig/Datasets/BEHAVE-dataset/objects'
        # mesh_path = os.path.join(obj_path, simplified_mesh[obj_name])
        # temp_simp = trimesh.load(mesh_path)
        # obj_points = temp_simp.vertices

        # # # center the meshes
        # center = np.mean(obj_points, 0)
        # obj_points -= center
        # obj_points = obj_points.astype(np.float32)

        # # Calculate the simple moving average
        # motion_obj = data_obj

        # angle, trans = motion_obj[:, :3].transpose(1, 0), motion_obj[:, 3:].transpose(1, 0)
        # rot = Rotation.from_rotvec(angle.transpose(1, 0)).as_matrix()
        # obj_points = np.matmul(obj_points[np.newaxis], rot.transpose(0, 2, 1)[:, np.newaxis])[:, 0] + trans.transpose(1, 0)[:, np.newaxis]
        # # obj_points = obj_points.transpose(1, 2, 0)

        # text_data = []
        # flag = False
        # with cs.open(pjoin('/work/vig/xiaogangp/codes/guided-motion-diffusion/dataset/Behave_global/texts', source_file[:-4] + '.txt')) as f:
        #     for line in f.readlines():
        #         text_dict = {}
        #         line_split = line.strip().split('#')
        #         caption = line_split[0]
        #         tokens = line_split[1].split(' ')
        #         # f_tag = float(line_split[2])
        #         # to_tag = float(line_split[3])
        #         # f_tag = 0.0 if np.isnan(f_tag) else f_tag
        #         # to_tag = 0.0 if np.isnan(to_tag) else to_tag
        #         # TODO: hardcode
        #         f_tag = to_tag = 0.0

        #         text_dict['caption'] = caption
        #         text_dict['tokens'] = tokens
        #         if f_tag == 0.0 and to_tag == 0.0:
        #             flag = True
        #             text_data.append(text_dict)

        # data = np.concatenate([rec_ric_data.reshape(-1, 22, 3), obj_points], 1)
        # plot('./check_videos/{}.mp4'.format(source_file[:-4]), data, None, None, title=str(text_data[0]['caption']), fps=20)

        np.save(pjoin(save_dir1, source_file), rec_ric_data)
        np.save(pjoin(save_dir2, source_file), data)
        frame_num += data.shape[0]

        # except Exception as e:
        #     print(f'Error in {source_file}: {e}')
    #         print(source_file)
    #         break
    #         import pdb; pdb.set_trace()

    print(
        "Total clips: %d, Frames: %d, Duration: %fm"
        % (len(source_list), frame_num, frame_num / 20 / 60)
    )
