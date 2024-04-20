import numpy as np
import os
import glob


def get_all_filenames(root_dir):
    file_names = glob.glob(root_dir + "*.npy")
    return file_names


def get_length_distribution(file_names):
    seq_lengths = []
    for i, file_path in enumerate(file_names):
        data = np.load(file_path, allow_pickle=True)
        seq_length = data.shape[1]
        seq_lengths.append(seq_length)

    return np.array(seq_lengths)


if __name__ == "__main__":
    root_dir = "./behave_t2m/new_joints_local/"
    file_names = get_all_filenames(root_dir)
    seq_lengths = get_length_distribution(file_names)
    print(
        f"Mean seq length: {round(np.mean(seq_lengths), 2)}, std. {round(np.std(seq_lengths), 2)}"
        f"Min: {np.min(seq_lengths)}, max {np.max(seq_lengths)}"
    )
