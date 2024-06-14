# DexHOI: End-to-End Learning of Full-Body Human-Object Interactions with Language Instructions

This is a repository for diffusion-based learning of full-body human-object interactions, conditioned on language instructions. 

Preprocessing the grab data is the first step towards training your model.

## GRAB Preprocessing

Create the preprocessing environment:

```bash
conda env create -f environment.yml
```

Alternatively:

```bash
conda create -n dexhoi python=3.7
conda activate dexhoi

# adjust to suitable CUDA version (10.x or 11.x)
# tested with torch 1.7.1, other versions might work too
pip install numpy==1.23.1 ipykernel skelvis trimesh matplotlib pyrender smplx

conda install numpy=1.23.1 pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install scipy pandas tqdm

cd common/manopth
pip install .
```

Download the SMPLX models and the MANO models. Create a folder `data_preprocessing/grab_preprocessing/body_models/mano` and place the MANO models in there. 

For HumanML3d preprocessing, create a folder `data_preprocessing/humanml3d/body_models/mano` and place the MANO models in there. Repeat the same for `smplh` and `dmpls` models, which you can download from the SMPL website. You should have `mano`, `smplh` and `dmpls` subfolders with models.

Then, download the GRAB zip files from the GRAB website. Unzip them using 
```bash
data_preprocessing/grab_preprocessing/GRAB/grab/unzip_grab.py --grab_path <path_to_grab_zips> --extract_path <path_to_output_dir>.
```
### Running the preprocessing

Once you have downloaded and unzipped the GRAB dataset, you are reasy to run the preprocessing. First, consider `data_preprocessing/grab_preprocessing/converter.ipynb`. Modify the paths to your needs, and execute all cells. 

Then, venture into `data_preprocessing/humanml3d/raw_pose_processing.ipynb`. In this file, you may change the hand representaion, if desired. Furthermore, it downsamples the data to your desired framerate and applies first transformations. Again, modify the paths to your needs and execute all cells. 

Then, go into `data_preprocessing/humanml3d/motion_representation.py`, again modifying the data paths to your needs and execute the Python file. If you change the hand dimensionality, make sure to modify it here as well. In this file, a local body motion representation is computed. The results are stored as `new_joints_local` and `new_joint_vecs_local`, which are the key files that contain our training data.

Finally, the last step is running `data_preprocessing/humanml3d/cal_mean_variance.py` after adjusting the paths in this file. Adjust the dimensionality as needed. This file simply computes a mean and variance for the training data.

This process will have created several directories and files, which you will need to compose into one directory that can be used for training in HOI-Diff. A typical training directory has the following structure:

```
- behave_t2m
    - new_joint_vecs_local
    - new_joints_local
    - texts
    - object_sample
    - object_mesh
    - split.json
    - Mean_local.npy
    - Std_local.npy
    - test.txt
    - train.txt
```
The preprocessing pipeline creates all of these files in the directories that are specified. You will need to copy all created files into a single directory, similar to the structure shown above. 

To train on the data, at the moment, copy the train data folder into `HOI-Diff/dataset/` and rename it to `behave_t2m` (regardless of whether the dataset contains BEHAVE data). This remains unchanged so far since proper naming convention would require a large refactoring across the HOI-Diff codebase, but can be changed in the future with more time.

## HOI-Diff

Create the environment: 

```bash
cd HOI-Diff
conda env create -f environment.yml
```
This has given errors with CUDA before. Alternatively, proceed with installations manually.

```bash
conda create -n t2hoi python=3.7
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

conda install numpy=1.12 scikit-learn
pip install ffmpeg-python chumpy smplx[all] h5py blobfile spacy
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git

cd common/manopth
pip install .
```

Then, follow the other steps from the [HOI-Diff setup](https://github.com/neu-vi/HOI-Diff) to download the missing dependencies. Additionally, in `HOI-Diff/body_models/`, add the MANO body model folder from the GRAB environment.

Verify GPU support:
```python
import torch
torch.cuda.is_available()
```

Download additional files:
```
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2hoi_evaluators.sh  
```

Train HOI-DM:
```bash
cd HOI-Diff
python -m train.hoi_diff --save_dir ./save/<example_run_name> --dataset behave --save_interval 20000 --num_steps 100000 --arch trans_en
c --batch_size 24 --overwrite
```

Depending on your hand representation, you need to search for the `HAND_MODE` variable in HOI-Diff, and change it and the `HAND_FEATURE_DIM` accordingly. In the future, this can be moved to a central configuration file.

You can view logs using:

```bash
cd HOI-Diff
tensorboard --log_dir ./save/<example_run_name>
```

Generate samples with HOI-DM:
```bash
python -m sample.local_generate_obj --model_path ./save/<example_model_run>/model000020000.pt --num_samples 10 --num_repetitions 1 --m
otion_length 10 --multi_backbone_split 4 --skip_first_stage
```

This command will generate samples. You can view the visualized output in the model save directory.


### Results Visualization (without hands)

First, install Blender 2.93:

```bash 
sudo snap install blender --channel=2.93lts/stable --classic\
```

Then, locate the Python installation of Blender:

```bash 
blender --background --python-expr "import sys; import os; print('\nThe path to the installation of python of blender can be:'); print('\n'.join(['- '+x.replace('/lib/python', '/bin/python') for x in sys.path if 'python' in (file:=os.path.split(x)[-1]) and not file.endswith('.zip')]))"
export BLENDER_PY = <the path to Blender Python>
```

Then, install the following dependencies:

```bash
$BLENDER_PY -m pip install --user numpy matplotlib moviepy shortuuid 
$BLENDER_PY -m pip install --user hydra-core --upgrade 
$BLENDER_PY -m pip install --user hydra_colorlog --upgrade 
$BLENDER_PY -m pip install -r HOI-Diff/blender_render/requirements.txt
```

Then, create the SMPL mesh from the skeletal joints, for example, using a command like this:

```python 
python -m visualize.render_mesh --input_path save/grab_30fps_noflipsatall_enc_512/samples_grab_30fps_noflipsatall_enc_512_000020000_see
d10/sample01_rep00.mp4
```
The files from this will show up in the base directory that you provided. Copy them into the newly created subdirectory `<sampleXX_repXX_obj/`.

Note that the video file you refer to has to have the underscore with `<repXX>`.  

Then, begin rendering with:

```bash
cd blender_render
blender --background --python render.py -- --dir /home/erik/ethz/digital-humans/dex-hoi/HOI-Diff/save/grab_30fps_noflipsatall_enc_512/samples_grab_30fps_noflipsatall_enc_512_000020000_seed10/sample01_rep00_obj/
```
