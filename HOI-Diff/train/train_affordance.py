# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
import torch
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from utils.model_util import create_model_and_diffusion, load_pretrained_mdm
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from model.afford_est import AffordEstimation
from diffusion.gaussian_diffusion import AffordDiffusion

def main():
    args = train_args()  # Parse command-line arguments or configurations for training.
    fixseed(args.seed) # Fix the random seed for reproducibility.

    train_platform_type = eval(args.train_platform_type) # Dynamically evaluate the training platform type (e.g., ClearML, Tensorboard).
    train_platform = train_platform_type(args.save_dir) # Initialize the training platform with a directory for saving outputs.
    train_platform.report_args(args, name='Args') # Log or report the parsed arguments for tracking and reproducibility.
    
    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.') # Ensure there's a save directory specified.
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir)) # Avoid overwriting an existing directory unless explicitly allowed.
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir) # Create the directory if it does not exist.
        
    args_path = os.path.join(args.save_dir, 'args.json') # Define path for saving arguments.
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True) # Save arguments in a readable JSON format for later reference.

    dist_util.setup_dist(args.device) # Setup distributed computing environments if required by the device configuration.

    print("creating data loader...")
    data_conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        training_stage=1,
        use_global=args.global_3d
    ) # Configure data loading specifics based on provided arguments.
    data = get_dataset_loader(data_conf) # Load the dataset based on the configuration above
    
    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data, ModelClass=AffordEstimation, DiffusionClass=AffordDiffusion) # Begin model and diffusion process creation

    model.to(dist_util.dev()) # Move the model to the appropriate device (CPU/GPU)

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0)) # Calculate and print the total number of trainable parameters in the model
    print("Training...") # Start the training process
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close() # Close the training platform post training

if __name__ == "__main__":
    main()
