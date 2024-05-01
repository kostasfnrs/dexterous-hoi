import numpy as np
import sys
import os
from os.path import join as pjoin


# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def mean_variance(data_dir, save_dir, joints_num):
    file_list = os.listdir(data_dir)
    data_list = []

    for file in file_list:
        data = np.load(pjoin(data_dir, file))
        if np.isnan(data).any():
            print(f'Ignoring {file} due to NaN values')
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)

    # NOTE: this is horrible, added some comments to make it a bit clearer

    # root rot velocity
    Std[0:1] = Std[0:1].mean() / 1.0
    
    # root linear velocity
    Std[1:3] = Std[1:3].mean() / 1.0
    
    # root y
    Std[3:4] = Std[3:4].mean() / 1.0
    
    # ric data: start from 4 previous, go to 4 + ric_data_dim = 4 + (joints_num - 1) * 3
    Std[4: 4+(joints_num - 1) * 3] = Std[4: 4 +(joints_num - 1) * 3].mean() / 1.0
    
    # rot data: start from previous end index, add rot_data_dim = (joints_num - 1) * 6 (this gets absorbed into 3+6=9)
    Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9] = Std[4 + (joints_num - 1) * 3: 4+(joints_num - 1) * 9].mean() / 1.0

    # local velocity: start from previous end index, add local_velocity_dim = joint_num*3
    Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3] = Std[4 + (joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3].mean() / 1.0
    
    # foot contact: start from previous end index, add foot_contact_dim = 4
    Std[4 + (joints_num - 1) * 9 + joints_num * 3: 4 + (joints_num - 1) * 9 + joints_num * 3 + 4] = Std[4 + (joints_num - 1) * 9 + joints_num * 3: 4 + (joints_num - 1) * 9 + joints_num * 3 + 4].mean() / 1.0
    
    # start from previous end index, add 3 more dimensions for object position?
    Std[4 + (joints_num - 1) * 9 + joints_num * 3 + 4: 4 + (joints_num - 1) * 9 + joints_num * 3 + 7] = Std[4 + (joints_num - 1) * 9 + joints_num * 3 + 4:4 + (joints_num - 1) * 9 + joints_num * 3 + 7].mean() / 1.0
    
    # final 3 dimensions for object velocity?
    # NOTE: updated with hands, so we take three for the velocity
    Std[4 + (joints_num - 1) * 9 + joints_num * 3 + 7:4 + (joints_num - 1) * 9 + joints_num * 3 + 10] = Std[4 + (joints_num - 1) * 9 + joints_num * 3 + 7:4 + (joints_num - 1) * 9 + joints_num * 3 + 10].mean() / 1.0

    # left hand, 24 dims
    Std[4 + (joints_num - 1) * 9 + joints_num * 3 + 10:4 + (joints_num - 1) * 9 + joints_num * 3 + 34] = Std[4 + (joints_num - 1) * 9 + joints_num * 3 + 10:4 + (joints_num - 1) * 9 + joints_num * 3 + 34].mean() / 1.0
    Std[4 + (joints_num - 1) * 9 + joints_num * 3 + 34:4 + (joints_num - 1) * 9 + joints_num * 3 + 58] = Std[4 + (joints_num - 1) * 9 + joints_num * 3 + 34:4 + (joints_num - 1) * 9 + joints_num * 3 + 58].mean() / 1.0
    assert 8 + (joints_num - 1) * 9 + joints_num * 3 + 6 + 2 * 24 == Std.shape[-1]

#     np.save(pjoin(save_dir, 'Mean.npy'), Mean)
#     np.save(pjoin(save_dir, 'Std.npy'), Std)

    np.save(pjoin(save_dir, 'Mean_local.npy'), Mean)
    np.save(pjoin(save_dir, 'Std_local.npy'), Std)

    return Mean, Std


if __name__ == '__main__':

    data_root = '/media/erik/DATA'
    
    #     data_dir = './HumanML3D/new_joint_vecs/'
    # data_dir = './data/new_joint_vecs_local/'
    data_dir = f'{data_root}/grab/new_joint_vecs_local/'
    save_dir = f'{data_root}/grab/calibration/'
    os.makedirs(save_dir, exist_ok=True)
    mean, std = mean_variance(data_dir, save_dir, 22)
#     print(mean)
#     print(Std)
