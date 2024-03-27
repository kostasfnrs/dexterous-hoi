#!/bin/bash
#SBATCH --chdir .
#SBATCH --account digital_humans
#SBATCH --time=48:00:00
#SBATCH -o /home/%u/hoi_diff_slurm_output__%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mem-per-cpu=14G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1 # Make sure this requests an NVIDIA A5000 GPU if needed; you might need to specify this differently.

set -e
set -o xtrace

echo "PWD: $(pwd)"
echo "STARTING AT $(date)"

# Environment setup
source /cluster/courses/digital_humans/datasets/team_2/miniconda/etc/profile.d/conda.sh
conda activate t2hoi # Activate your specific conda environment

# Pre-check CUDA availability
python -c "import torch; print('Cuda available?', torch.cuda.is_available())"

# Run your training command
python -m train.hoi_diff --save_dir ./save/behave_enc_512 --dataset behave --save_interval 1000 --num_steps 20000 --arch trans_enc --batch_size 32

echo "Done."
echo "FINISHED at $(date)"
