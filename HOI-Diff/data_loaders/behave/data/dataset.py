import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy
from torch.utils.data._utils.collate import default_collate
from data_loaders.behave.utils.word_vectorizer import WordVectorizer
from data_loaders.behave.utils.get_opt import get_opt
from visualize.vis_utils import simplified_mesh
import trimesh
from scipy.spatial.transform import Rotation
from data_loaders.behave.scripts.motion_process import (
    recover_from_ric,
    extract_features,
)
import scipy.sparse
from data_loaders.behave.utils.paramUtil import *
from utils.utils import recover_obj_points
from data_loaders.behave.utils.plot_script import plot_3d_motion

HAND_MODE = "PCA"
if HAND_MODE == "PCA":
    HAND_FEATURE_DIM = 27
elif HAND_MODE == "joints":
    HAND_FEATURE_DIM = 63


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


"""For use of training text motion matching model, and evaluations"""


class Text2AffordDataset(data.Dataset):
    def __init__(self, opt, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.pointer = 0

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                # load obj points----------------
                obj_name = name.split("_")[2]
                obj_path = pjoin(opt.data_root, "object_mesh")
                mesh_path = os.path.join(obj_path, simplified_mesh[obj_name])
                temp_simp = trimesh.load(mesh_path)

                obj_points = np.array(temp_simp.vertices)
                obj_faces = np.array(temp_simp.faces)

                # center the meshes
                center = np.mean(obj_points, 0)
                obj_points -= center
                obj_points = obj_points.astype(np.float32)

                # sample object points
                obj_sample_path = pjoin(
                    opt.data_root, "object_sample/{}.npy".format(name)
                )
                o_choose = np.load(obj_sample_path)

                obj_points = obj_points[o_choose]

                contact_input = np.load(
                    pjoin(opt.data_root, "affordance_data/contact_" + name + ".npy"),
                    allow_pickle=True,
                )[None][0]

                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        # f_tag = float(line_split[2])
                        # to_tag = float(line_split[3])
                        # f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        # to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        # TODO: hardcode
                        f_tag = to_tag = 0.0

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20) : int(to_tag * 20)]
                                if (len(n_motion)) < min_motion_len or (
                                    len(n_motion) >= 200
                                ):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                    + "_"
                                    + name
                                )
                                while new_name in data_dict:
                                    new_name = (
                                        random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                        + "_"
                                        + name
                                    )
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "contact_input": contact_input,
                        "text": text_data,
                        "seq_name": name,
                        "obj_points": obj_points,
                    }

                    new_name_list.append(name)
                    # length_list.append(len(motion))
                    # TODO: harcode
                    # for i in range(1000):
                    #     data_dict[name+"_{}".format(i)] = data_dict[name]
                    #     new_name_list.append(name+"_{}".format(i))
                    #     length_list.append(len(motion))
            except:
                pass

        name_list = sorted(new_name_list, key=lambda x: x[1])

        self.data_dict = data_dict
        self.name_list = name_list

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        contact_input, text_list, seq_name, obj_points = (
            data["contact_input"],
            data["text"],
            data["seq_name"],
            data["obj_points"],
        )
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        return (
            None,
            None,
            caption,
            None,
            contact_input,
            "_".join(tokens),
            seq_name,
            obj_points,
        )


class TextOnlyDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        # more length limiting for some reason - disable this
        # self.mean = mean[: opt.dim_pose]
        # self.std = std[: opt.dim_pose]
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.max_length = 20
        self.pointer = 0
        self.fixed_length = 196
        self.normal_dim = opt.dim_pose

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        self.id_list = id_list

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                # load obj points----------------
                # obj_name = name.split('_')[2]
                obj_name = name.split("_")[1]
                obj_path = pjoin(opt.data_root, "object_mesh")
                # mesh_path = os.path.join(obj_path, simplified_mesh[obj_name])
                mesh_path = os.path.join(obj_path, obj_name, f"{obj_name}.ply")

                temp_simp = trimesh.load(mesh_path)
                obj_points = np.array(temp_simp.vertices).astype(np.float32)
                obj_faces = np.array(temp_simp.faces).astype(np.float32)

                # sample object points
                obj_sample_path = pjoin(
                    opt.data_root, "object_sample/{}.npy".format(name)
                )
                o_choose = np.load(obj_sample_path)

                # contact_input = np.load(pjoin(opt.data_root, 'affordance_data/contact_'+name + '.npy'), allow_pickle=True)[None][0]

                # center the meshes
                center = np.mean(obj_points, 0)
                obj_points -= center

                obj_points = obj_points[o_choose]
                obj_normals = obj_faces[o_choose]

                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        # TODO: hardcode
                        # f_tag = to_tag = 0.0

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                    + "_"
                                    + name
                                )
                                while new_name in data_dict:
                                    new_name = (
                                        random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                        + "_"
                                        + name
                                    )
                                data_dict[new_name] = {
                                    "text": [text_dict],
                                    "seq_name": name,
                                    "obj_points": obj_points,
                                    "obj_normals": obj_normals,
                                }
                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "text": text_data,
                        "seq_name": name,
                        "obj_points": obj_points,
                        "obj_normals": obj_normals,
                    }
                    new_name_list.append(name)
            except FileNotFoundError as err:
                print(err.__class__.__name__)
                print(err)
                print(f"Error in {name}, possibly did not find text annotation?")
                pass
            except IndexError as err:
                print(err.__class__.__name__)
                print(err)
                print(f"Error in {name}, error in text annotation?")
                pass
            except Exception as err:
                print(err.__class__.__name__)
                print(err)
                print(f"Error in {name}")
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        data = data.clone()
        if data.shape[-1] == 269 + 2 * HAND_FEATURE_DIM:
            assert (
                len(self.mean) == 269 + 2 * HAND_FEATURE_DIM
            ), f"Mean feature dim {len(self.mean)} is not matching"
            assert (
                len(self.std) == 269 + 2 * HAND_FEATURE_DIM
            ), f"Std feature dim {len(self.std)} is not matching"
            data = data * self.std + self.mean
        else:
            data[..., :263] = data[..., :263] * self.std[:263] + self.mean[:263]
        return data

    def inv_transform_th(self, data):
        data = data * torch.from_numpy(self.std).to(data.device) + torch.from_numpy(
            self.mean
        ).to(data.device)
        return data

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]

        text_list, seq_name, obj_points, obj_normals = (
            data["text"],
            data["seq_name"],
            data["obj_points"],
            data["obj_normals"],
        )
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        return (
            None,
            None,
            caption,
            None,
            np.array([0]),
            self.fixed_length,
            None,
            seq_name,
            obj_points,
            obj_normals,
        )
        # fixed_length can be set from outside before sampling


"""For use of training text motion matching model, and evaluations"""


class Text2MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40
        self.normal_dim = opt.dim_pose

        data_dict = {}
        id_list = []

        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        print(f"Length of id_list: {len(id_list)}")
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))

                # load obj points----------------
                # NOTE: this is so hacky and hardcoded, needs to be changed for every dataset
                obj_name = name.split("_")[1]
                obj_path = pjoin(opt.data_root, "object_mesh")
                # mesh_path = os.path.join(obj_path, simplified_mesh[obj_name])
                mesh_path = os.path.join(obj_path, obj_name, f"{obj_name}.ply")
                temp_simp = trimesh.load(mesh_path)

                obj_points = np.array(temp_simp.vertices)
                obj_faces = np.array(temp_simp.faces)

                # center the meshes
                center = np.mean(obj_points, 0)
                obj_points -= center
                obj_points = obj_points.astype(np.float32)

                # sample object points
                obj_sample_path = pjoin(
                    opt.data_root, "object_sample/{}.npy".format(name)
                )
                o_choose = np.load(obj_sample_path)

                obj_points = obj_points[o_choose]
                obj_normals = obj_faces[o_choose]

                # TODO: hardcode
                motion = motion[:199].astype(np.float32)

                # contact_input = np.load(pjoin(opt.data_root, 'affordance_data/contact_'+name + '.npy'), allow_pickle=True)[None][0]

                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    print("skipping %s due to wrong motion length" % name)
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20) : int(to_tag * 20)]
                                if (len(n_motion)) < min_motion_len or (
                                    len(n_motion) >= 200
                                ):
                                    continue
                                # new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name

                                data_dict[name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                    "seq_name": name,
                                    "obj_points": obj_points,
                                    "obj_normals": obj_normals,
                                    # 'gt_afford_labels': contact_input
                                }
                                new_name_list.append(name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                        "seq_name": name,
                        "obj_points": obj_points,
                        "obj_normals": obj_normals,
                        # 'gt_afford_labels':contact_input
                    }

                    new_name_list.append(name)
                    length_list.append(len(motion))
            except FileNotFoundError as err:
                print(err.__class__.__name__)
                print(err)
                print(f"Error in {name}, possibly did not find text annotation?")
                pass
            except IndexError as err:
                print(err.__class__.__name__)
                print(err)
                print(f"Error in {name}, error in text annotation?")
                pass
            except Exception as err:
                print(err.__class__.__name__)
                print(err)
                print(f"Error in {name}")
                pass

        print(f"Length of new name list: {len(new_name_list)}")
        print(f"Length of length list: {len(length_list)}")
        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1])
        )

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        data = data * self.std[: data.shape[-1]] + self.mean[: data.shape[-1]]
        return data

    def inv_transform_th(self, data):
        data = data * torch.from_numpy(self.std).to(data.device) + torch.from_numpy(
            self.mean
        ).to(data.device)
        return data

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list, seq_name, obj_points, obj_normals = (
            data["motion"],
            data["length"],
            data["text"],
            data["seq_name"],
            data["obj_points"],
            data["obj_normals"],
        )

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[: self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            if len(token.split("/")) < 2:
                print(f" {seq_name}   {tokens}")
                break
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"
        if coin2 == "double":
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx : idx + m_length]

        if not self.opt.use_global:
            "Z Normalization"
            motion = np.copy(motion)
            if len(self.mean) == 269 + 2 * HAND_FEATURE_DIM:
                np.seterr(all="raise")
                # print(f'Length of batch: {motion.shape[0]}')
                # print(f'Std. deviation: {np.mean(self.std)}')
                try:
                    end_idx = 269 + 2 * HAND_FEATURE_DIM
                    motion[:, :end_idx] = (
                        motion[:, :end_idx] - self.mean[:end_idx]
                    ) / self.std[:end_idx]
                except FloatingPointError as err:
                    print(err.__class__.__name__)
                    print(err)
                    print(f"Error in {seq_name}, possibly due to zero division?")
                    print(f"Std. deviation: {np.where(self.std==0)}")
            else:
                #  for evaluation of ground truth
                motion[..., :263] = (motion[..., :263] - self.mean[:263]) / self.std[
                    :263
                ]

        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [
                    motion,
                    np.zeros((self.max_motion_length - m_length, motion.shape[1])),
                ],
                axis=0,
            )

        # Contact labels here for evaluation!
        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
            seq_name,
            obj_points,
            obj_normals,
        )


# A wrapper class for behave dataset t2m and t2afford
class Behave(data.Dataset):
    def __init__(
        self,
        mode,
        datapath="./dataset/behave_opt.txt",
        split="train",
        use_global=False,
        training_stage=1,
        wo_obj_motion=False,
        **kwargs,
    ):
        self.mode = mode

        self.dataset_name = "t2m_behave"
        self.dataname = "t2m_behave"

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f"."
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = (
            None  # torch.device('cuda:4') # This param is not in use in this context
        )
        opt = get_opt(dataset_opt_path, device, use_global, wo_obj_motion)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.afford_dir = pjoin(abs_base_path, opt.afford_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = "./dataset"
        self.opt = opt
        self.use_global = use_global
        self.training_stage = training_stage
        print("Loading dataset %s ..." % opt.dataset_name)

        if self.training_stage == 1:
            self.split_file = pjoin(
                opt.data_root, f"{split}.txt"
            )  #   adopt augmented data for affordance training
            if mode == "text_only":
                self.t2m_dataset = TextOnlyAffordDataset(self.opt, self.split_file)
            else:
                self.w_vectorizer = WordVectorizer(
                    pjoin(abs_base_path, "glove"), "our_vab"
                )
                self.t2m_dataset = Text2AffordDataset(
                    self.opt, self.split_file, self.w_vectorizer
                )

        elif self.training_stage == 2:
            if mode == "gt":
                print("Loading ground truth mean and std")
                # used by T2M models (including evaluators)
                self.mean = np.load(pjoin(opt.meta_dir, f"t2m_mean.npy"))
                self.std = np.load(pjoin(opt.meta_dir, f"t2m_std.npy"))

            elif mode in ["train", "eval", "text_only"]:
                # used by our models
                print(f"Loading local mean and std from {opt.data_root}")
                self.mean = np.load(pjoin(opt.data_root, "Mean_local.npy"))
                self.std = np.load(pjoin(opt.data_root, "Std_local.npy"))
                print(f"Mean: {self.mean.shape}  Std: {self.std.shape}")

            if mode == "eval":
                # used by T2M models (including evaluators)
                # this is to translate their norms to ours
                print("Loading eval mean and std")

                self.mean_for_eval = np.load(pjoin(opt.meta_dir, f"t2m_mean.npy"))
                self.std_for_eval = np.load(pjoin(opt.meta_dir, f"t2m_std.npy"))

            self.split_file = pjoin(opt.data_root, f"{split}.txt")
            # self.split_file = pjoin('/work/vig/xiaogangp/codes/hoi-motion', f'{split}.txt')
            if mode == "text_only":
                self.t2m_dataset = TextOnlyDataset(
                    self.opt, self.mean, self.std, self.split_file
                )
            else:
                self.w_vectorizer = WordVectorizer(
                    pjoin(abs_base_path, "glove"), "our_vab"
                )
                self.t2m_dataset = Text2MotionDatasetV2(
                    self.opt, self.mean, self.std, self.split_file, self.w_vectorizer
                )
                self.num_actions = 1  # dummy placeholder

        else:
            print(f"error!")

        assert len(self.t2m_dataset) > 1, (
            "You loaded an empty dataset, "
            "it is probably because your data dir has only texts and no motions.\n"
            "To train and evaluate MDM you should get the FULL data as described "
            "in the README file."
        )

        # Load necessay variables for converting raw motion to processed data
        data_dir = "./dataset/000021.npy"
        self.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
        self.kinematic_chain = t2m_kinematic_chain
        # # Get offsets of target skeleton
        # example_data = np.load(data_dir)
        # example_data = example_data.reshape(len(example_data), -1, 3)
        # example_data = torch.from_numpy(example_data)
        # tgt_skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, 'cpu')
        # # (joints_num, 3)
        # tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()

    def motion_to_rel_data(self, motion, model, is_norm=False):
        motion_bu = motion.detach().clone()
        # Right/Left foot
        fid_r, fid_l = [8, 11], [7, 10]
        # Face direction, r_hip, l_hip, sdr_r, sdr_l
        face_joint_indx = [2, 1, 17, 16]
        sample_rel_np_list = []
        for ii in range(len(motion)):
            # Data need to be [120 (timestep), 22, 3] to get feature
            sample_rel = extract_features(
                motion[ii].detach().cpu().clone().permute(2, 0, 1).cpu().numpy(),
                0.002,
                self.n_raw_offsets,
                self.kinematic_chain,
                face_joint_indx,
                fid_r,
                fid_l,
            )
            # Duplicate last motion step to match the size
            sample_rel = torch.from_numpy(sample_rel).unsqueeze(0).float()
            # sample_rel = torch.cat(
            #     [sample_rel, sample_rel[0:1, -1:, :].clone()], dim=1)

            # Normalize with relative normalization
            if is_norm:
                sample_rel = (sample_rel - self.mean_rel[:263]) / self.std_rel[:263]
            sample_rel = sample_rel.unsqueeze(1).permute(0, 3, 1, 2)
            sample_rel = sample_rel.to(motion.device)
            sample_rel_np_list.append(sample_rel)

        processed_data = torch.cat(sample_rel_np_list, axis=0)

        n_joints = 22
        # NOTE: check if the sequence is still that same after extract_features and converting back
        # sample = dataset.t2m_dataset.inv_transform(sample_abs.cpu().permute(0, 2, 3, 1)).float()
        # sample_after = (processed_data.permute(0, 2, 3, 1) * self.std_rel) + self.mean_rel

        # print(f"processed_data:{processed_data.shape}  {sample_after.shape}")
        # B, _, T , F = sample_after.shape
        # sample_after = sample_after[..., :66].reshape(B, T, n_joints, 3).permute(0,2,3,1)

        # sample_after = recover_from_ric(sample_after, n_joints)
        # sample_after = sample_after.view(-1, *sample_after.shape[2:]).permute(0, 2, 3, 1)

        # rot2xyz_pose_rep = 'xyz'
        # rot2xyz_mask = None
        # sample_after = model.rot2xyz(x=sample_after,
        #                     mask=rot2xyz_mask,
        #                     pose_rep=rot2xyz_pose_rep,
        #                     glob=True,
        #                     translation=True,
        #                     jointstype='smpl',
        #                     vertstrans=True,
        #                     betas=None,
        #                     beta=0,
        #                     glob_rot=None,
        #                     get_rotations_back=False)

        # from data_loaders.humanml.utils.plot_script import plot_3d_motion

        # for i in range(motion.shape[0]):
        #     # print(f"test:{ sample_after.shape}   {motion[2].permute(2,0,1).shape}")
        #     plot_3d_motion("./test_positions_{}.mp4".format(i), self.kinematic_chain, motion[i].permute(2,0,1).detach().cpu().numpy(), 'title', 'humanml', fps=20)
        #     plot_3d_motion("./test_positions_1_after{}.mp4".format(i), self.kinematic_chain, sample_after[i].permute(2,0,1).detach().cpu().numpy(), 'title', 'humanml', fps=20)

        # Return data already normalized with relative mean and std. shape [bs, 263, 1, 120(motion step)]
        return processed_data


def text_to_object(text):
    obj_list = [
        # NOTE: there is apparently a hand object, which causes bugs for the object identification here
        # "hand",
        "camera",
        "cylinder_small",
        "knife",
        "toothbrush",
        "utah_teapot",
        "ps_controller",
        "elephant",
        "cup",
        "scissors",
        "cube_large",
        "mug",
        "stapler",
        "torus_small",
        "pyramid_medium",
        "piggy_bank",
        "wristwatch",
        "light_bulb",
        "stanford_bunny",
        "apple",
        "torus_medium",
        "bowl",
        "hammer",
        "torus_large",
        "cylinder_medium",
        "mouse",
        "banana",
        "headphones",
        "pyramid_large",
        "binoculars",
        "door_knob",
        "cube_small",
        "sphere_small",
        "water_bottle",
        "rubber_duck",
        "train",
        "cube_medium",
        "wine_glass",
        "sphere_large",
        "cylinder_large",
        "eyeglasses",
        "airplane",
        "pan",
        "cell_phone",
        "pyramid_small",
        "alarm_clock",
        "toothpaste",
        "sphere_medium",
        "flashlight",
        "flute",
    ]

    all_obj_points = []
    all_obj_normals = []
    all_obj_names = []
    import re

    for i in range(len(text)):
        name = None
        for j in range(len(obj_list)):
            if obj_list[j] in text[i]:
                name = obj_list[j]
                break

        assert name is not None, "Object name not found in text prompt"
        # load obj points----------------
        # obj_path = "/work/vig/xiaogangp/codes/hoi-motion_pretrained/object_mesh"
        obj_path = "./dataset/behave_t2m/object_mesh/"
        obj_name = name
        mesh_path = os.path.join(obj_path, obj_name, obj_name + ".ply")

        temp_simp = trimesh.load(mesh_path)
        obj_points = np.array(temp_simp.vertices).astype(np.float32) * 0.15
        obj_faces = np.array(temp_simp.faces).astype(np.float32)
        obj_normals = obj_faces

        # # sample object points
        # obj_sample_path = '/work/vig/yimingx/behave_obj_sample/{}.npy'.format(name)
        # o_choose = np.load(obj_sample_path)

        # # compute normal
        # mesh = o3d.geometry.TriangleMesh()
        # verts_full_new = obj_points
        # mesh.vertices = o3d.utility.Vector3dVector(verts_full_new)
        # mesh.triangles = o3d.utility.Vector3iVector(obj_faces)
        # mesh.compute_vertex_normals()
        # obj_normals = np.asarray(mesh.vertex_normals)
        # obj_points = np.asarray(mesh.vertices).astype(np.float32)

        # center the meshes
        center = np.mean(obj_points, 0)
        obj_points -= center

        replace = False if len(obj_points) > 512 else True
        choose = np.random.choice(len(obj_points), 512, replace=replace)
        choose = np.sort(choose)
        # np.save(obj_sample_path, choose)

        obj_points = obj_points[choose]
        obj_normals = obj_normals[choose]

        all_obj_points.append(obj_points)
        all_obj_normals.append(obj_normals)
        all_obj_names.append(obj_name)

    return np.array(all_obj_points), np.array(all_obj_normals), np.array(all_obj_names)


def sample_to_motion(sample_global, dataset, model):
    n_joints = 22
    # (bs, 262, 1, 120)
    # In case of random projection, this already includes undoing the random projection

    sample = dataset.t2m_dataset.inv_transform(
        sample_global.cpu().permute(0, 2, 3, 1)
    ).float()

    B, _, T, F = sample.shape
    sample = sample[..., :66].reshape(B, T, n_joints, 3).permute(0, 2, 3, 1)

    return sample


def global3d_to_rel(sample_global, dataset, model, is_norm=True):
    """We want to change the first 3 values from absolute to relative
    sample_abs shape [bs, 263, 1, 196]   [bs, 1,193, 263]
    """
    n_joints = 22
    # (bs, 263, 1, 120)
    # In case of random projection, this already includes undoing the random projection
    sample = dataset.t2m_dataset.inv_transform(
        sample_global.cpu().permute(0, 2, 3, 1)
    ).float()

    B, _, T, F = sample.shape
    sample_human = sample[..., :66].reshape(B, T, n_joints, 3).permute(0, 2, 3, 1)

    # Now convert skeleton back to sample with relative representation
    sample_rel = dataset.motion_to_rel_data(sample_human, model, is_norm=is_norm)
    sample_obj = sample.permute(0, 3, 1, 2)[:, -6:, :, :-1]

    return sample_rel, sample_obj
