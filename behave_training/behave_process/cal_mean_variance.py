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
            print(file)
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0
    Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9] = Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9].mean() / 1.0
    Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3] = Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3].mean() / 1.0
    Std[4 + (joints_num - 1) * 9 + joints_num * 3: 4 + (joints_num - 1) * 9 + joints_num * 3 +4 ] = Std[4 + (joints_num - 1) * 9 + joints_num * 3: 4 + (joints_num - 1) * 9 + joints_num * 3 + 4].mean() / 1.0
    Std[4 + (joints_num - 1) * 9 + joints_num * 3 +4 : 4 + (joints_num - 1) * 9 + joints_num * 3 + 7] = Std[4 + (joints_num - 1) * 9 + joints_num * 3 +4 :4 + (joints_num - 1) * 9 + joints_num * 3 +7 ].mean() / 1.0
    Std[4 + (joints_num - 1) * 9 + joints_num * 3 +7 : ] = Std[4 + (joints_num - 1) * 9 + joints_num * 3 +7 : ].mean() / 1.0
    assert 8 + (joints_num - 1) * 9 + joints_num * 3 + 6 == Std.shape[-1]

#     np.save(pjoin(save_dir, 'Mean.npy'), Mean)
#     np.save(pjoin(save_dir, 'Std.npy'), Std)
    
    np.save(pjoin(save_dir, 'Mean_local.npy'), Mean)
    np.save(pjoin(save_dir, 'Std_local.npy'), Std)

    return Mean, Std


if __name__ == '__main__':
#     data_dir = './HumanML3D/new_joint_vecs/'
    data_dir = './new_joint_vecs_local/'
    save_dir = './'
    mean, std = mean_variance(data_dir, save_dir, 22)
#     print(mean)
#     print(Std)