import os
import numpy as np
import torch
import shutil
from utils.fixseed import fixseed
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip, load_model
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from data_loaders.behave.scripts.motion_process import recover_from_ric
import data_loaders.behave.utils.paramUtil as paramUtil
from data_loaders.behave.utils.plot_script import plot_3d_motion
from data_loaders.tensors import collate
from scipy import linalg
from model.hoi_diff import HOIDiff as used_model
from diffusion.gaussian_diffusion import LocalMotionDiffusion
from sample.generate_hand_joints import generate_hand_joints

HAND_MODE = "PCA"
if HAND_MODE == "PCA":
    HAND_FEATURE_DIM = 27
elif HAND_MODE == "joints":
    HAND_FEATURE_DIM = 63

def load_ground_truth_data(joints_dir):
    joints_files = sorted([os.path.join(joints_dir, f) for f in os.listdir(joints_dir) if f.endswith('.npy')])
    all_joints = []
    max_joints_size = 0
    for jfile in joints_files:
        joints = np.load(jfile)
        max_joints_size = max(max_joints_size, joints.shape[1])
    for jfile in joints_files:
        joints = np.load(jfile)
        padded_joints = np.pad(joints, ((0, 0), (0, max_joints_size - joints.shape[1]), (0, 0), (0, 0)), 'constant')
        all_joints.append(padded_joints)
    all_joints = np.concatenate(all_joints, axis=0)
    return all_joints

def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def calculate_diversity(activations, diversity_times=200):
    num_motions = len(activations)
    diversity = 0
    first_indices = np.random.randint(0, num_motions, diversity_times)
    second_indices = np.random.randint(0, num_motions, diversity_times)
    for first_idx, second_idx in zip(first_indices, second_indices):
        diversity += np.linalg.norm(activations[first_idx] - activations[second_idx])
    diversity /= diversity_times
    return diversity

def calculate_multimodality(activations, labels, num_labels=29, multimodality_times=20):
    num_motions = len(labels)
    multimodality = 0
    label_quotas = np.repeat(multimodality_times, num_labels)
    count = 0
    while np.any(label_quotas > 0) and count <= 10000:
        count += 1
        first_idx = np.random.randint(0, num_motions)
        first_label = np.argmax(labels[first_idx])
        if first_label >= num_labels:  # Ensure label is within valid range
            continue
        if label_quotas[first_label] <= 0:
            continue
        second_idx = np.random.randint(0, num_motions)
        second_label = np.argmax(labels[second_idx])
        while first_label != second_label:
            second_idx = np.random.randint(0, num_motions)
            second_label = np.argmax(labels[second_idx])
        if second_label >= num_labels:  # Ensure label is within valid range
            continue
        label_quotas[first_label] -= 1
        multimodality += np.linalg.norm(activations[first_idx] - activations[second_idx])
    multimodality /= (multimodality_times * num_labels)
    return multimodality

def calc_AVE(joints_sbj, joints_gt):
    joints_sbj = torch.tensor(joints_sbj)
    joints_gt = torch.tensor(joints_gt)
    T = joints_gt.shape[0]
    J = joints_gt.shape[1]
    var_gt = torch.var(joints_gt, dim=0, unbiased=False)
    var_pred = torch.var(joints_sbj, dim=0, unbiased=False)
    mean_ave_loss = torch.mean((var_gt - var_pred) ** 2)*1000
    return mean_ave_loss

def calculate_mpjpe(gt_joints, pred_joints):
    assert gt_joints.shape == pred_joints.shape, "Shape mismatch between ground truth and predicted joints"
    mpjpe = np.mean(np.linalg.norm(gt_joints - pred_joints, axis=-1))  # Mean per joint position error
    return mpjpe

def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
    max_frames = 196 if args.dataset in ["kit", "humanml", "behave"] else 120
    fps = 20
    n_frames = min(max_frames, int(args.motion_length * fps))
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist(args.device)

    sigmoid = torch.nn.Sigmoid()
    if out_path == "":
        out_path = os.path.join(os.path.dirname(args.model_path),
                                "samples_{}_{}_seed{}".format(name, niter, args.seed))
        if args.text_prompt != "":
            out_path += "_" + args.text_prompt.replace(" ", "_").replace(".", "")
        elif args.input_text != "":
            out_path += "_" + os.path.basename(args.input_text).replace(".txt", "").replace(" ", "_").replace(".", "")

    # Load ground truth data
    joints_dir = '/cluster/courses/digital_humans/datasets/team_2/behave_t2m/new_joints_local'
    gt_joints = load_ground_truth_data(joints_dir)

    total_num_samples = args.num_samples * args.num_repetitions
    batch_size = args.batch_size
    num_batches = (total_num_samples + batch_size - 1) // batch_size

    print("Loading Motion dataset...")
    data = load_motion_dataset(args, max_frames, n_frames)

    print("Creating motion model and diffusion...")
    motion_model, motion_diffusion = load_model(
        args,
        data,
        dist_util.dev(),
        ModelClass=used_model,
        DiffusionClass=LocalMotionDiffusion,
        diff_steps=100,
        model_path=args.model_path,
    )

    if is_using_data:
        iterator = iter(data)
        _, model_kwargs = next(iterator)
    else:
        n_frames = int(args.motion_length * fps)
        collate_args = [
            {"inp": torch.zeros(n_frames), "tokens": None, "lengths": n_frames}
        ] * args.num_samples
        is_t2m = any([args.input_text, args.text_prompt])
        if is_t2m:
            obj_points, obj_normals, obj_name = text_to_object(texts)
            collate_args = [
                dict(arg, text=txt, obj_points=points, obj_normals=normals, seq_name=name)
                for arg, txt, points, normals, name in zip(collate_args, texts, obj_points, obj_normals, obj_name)
            ]
        _, model_kwargs = collate(collate_args)

    model_kwargs["y"]["obj_points"] = model_kwargs["y"]["obj_points"].to(dist_util.dev())
    model_kwargs["y"]["obj_normals"] = model_kwargs["y"]["obj_normals"].to(dist_util.dev())

    all_motions = []
    all_lengths = []
    all_obj_name = []

    for rep_i in range(args.num_repetitions):
        print(f"### Sampling [repetitions #{rep_i}]")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_num_samples)
            current_batch_size = end_idx - start_idx

            if args.guidance_param != 1:
                model_kwargs["y"]["scale"] = (
                    torch.ones(current_batch_size, device=dist_util.dev())
                    * args.guidance_param
                )

            sample_fn = motion_diffusion.p_sample_loop

            sample = sample_fn(
                motion_model,
                (
                    current_batch_size,
                    motion_model.njoints + 6 + 2 * HAND_FEATURE_DIM,
                    motion_model.nfeats,
                    n_frames,
                ),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
                cond_fn=None,
            )

            sample = data.dataset.t2m_dataset.inv_transform(
                sample.cpu().permute(0, 2, 3, 1)
            ).float()

            sample = sample[..., :263]
            n_joints = 22

            sample = recover_from_ric(sample, n_joints)
            sample = sample[:, :, :, : n_joints * 3]
            sample = sample.reshape(
                sample.shape[0], sample.shape[1], sample.shape[2], n_joints, 3
            )
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            all_motions.append(sample.cpu().numpy())
            all_lengths.append(model_kwargs["y"]["lengths"].cpu().numpy())
            all_obj_name += model_kwargs["y"]["seq_name"]

            print(f"created {len(all_motions) * batch_size} samples")

    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    all_obj_name = all_obj_name[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, "results.npy")

    save_dict = {
        "motion": all_motions,
        "lengths": all_lengths,
        "num_samples": args.num_samples,
        "num_repetitions": args.num_repetitions,
        "obj_name": all_obj_name,
    }

    print(f"saving results file to [{npy_path}]")
    np.save(npy_path, save_dict)

    with open(npy_path.replace(".npy", ".txt"), "w") as fw:
        fw.write("\n".join(all_obj_name))
    with open(npy_path.replace(".npy", "_len.txt"), "w") as fw:
        fw.write("\n".join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")

    def load_ground_truth_data_found(joints_dir, base_filename, n_frames):
        joints_path = os.path.join(joints_dir, f"{base_filename}.npy")
        
        if os.path.exists(joints_path):
            gt_joints = np.load(joints_path)
            if gt_joints.shape[1] > n_frames:
                gt_joints = gt_joints[:, :n_frames, ...]
            else:
                padding_shape_joints = ((0, 0), (0, n_frames - gt_joints.shape[1]), (0, 0), (0, 0))
                gt_joints = np.pad(gt_joints, padding_shape_joints, mode='constant')
            return gt_joints
        else:
            print(base_filename, "does not exist")
            return None

    matched_data = []
    for base_filename in save_dict['obj_name']:
        gt_joints = load_ground_truth_data_found(joints_dir, base_filename, n_frames)
        if gt_joints is not None:
            if not matched_data:
                matched_data.append({
                    'obj_name': base_filename,
                    'gt_joints': gt_joints[np.newaxis, ...],
                })
            else:
                matched_data[0]['gt_joints'] = np.concatenate((matched_data[0]['gt_joints'], gt_joints[np.newaxis, ...]), axis=0)
        else:
            print(f"Warning: .npy files not found for {base_filename}")

    if matched_data:
        print("Shape of gt_joints:", matched_data[0]['gt_joints'].shape)

    gt_joints = matched_data[0]['gt_joints']
    gt_joints = gt_joints[:,0,:,:,:]
    print(gt_joints.shape)

    joint_pred = save_dict["motion"]
    joint_pred = joint_pred.transpose(0,3,1,2)
    print(joint_pred.shape)

    gt_joints_flat = gt_joints.reshape(-1, 22, 3)
    joint_pred_flat = joint_pred.reshape(-1, 22, 3)

    gt_joints_flat = gt_joints_flat.reshape(gt_joints_flat.shape[0], -1)
    joint_pred_flat = joint_pred_flat.reshape(joint_pred_flat.shape[0], -1)

    mu1, sigma1 = calculate_activation_statistics(gt_joints_flat)
    mu2, sigma2 = calculate_activation_statistics(joint_pred_flat)

    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f'FID: {fid}')

    ave_loss = calc_AVE(joint_pred_flat, gt_joints_flat)
    print(f'AVE: {ave_loss}')

    # Calculate diversity
    diversity = calculate_diversity(joint_pred_flat)
    print(f'Diversity: {diversity}')

    # Calculate multimodality
    multimodality = calculate_multimodality(joint_pred_flat, gt_joints_flat)
    print(f'Multimodality: {multimodality}')

    # Calculate MPJPE
    mpjpe = calculate_mpjpe(gt_joints, joint_pred)
    print(f'Mean Per-Joint Positional Error (MPJPE): {mpjpe}')

def load_motion_dataset(args, max_frames, n_frames, training_stage=2):
    data_conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=max_frames,
        split="test",
        hml_mode="text_only",
        training_stage=training_stage,
    )
    data = get_dataset_loader(data_conf)
    data.fixed_length = n_frames
    return data

if __name__ == "__main__":
    main()
