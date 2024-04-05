# Language-conditioned HOI generation using the GRAB dataset

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

cd ./data_preprocessing/grab_preprocessing/manopth
pip install .

```

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
```

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

The preprocessed BEHAVE dataset from HOI-Diff is incomplete, use the dataset from the cluster:
```bash
scp -r <username>@student-cluster.inf.ethz.ch:/cluster/courses/digital_humans/datasets/team_2/new_hoi_diff/HOI-Diff/dataset/behave_t2m HOI-Diff/dataset/behave_t2m/
```

Train HOI-DM:
```bash
python -m train.hoi_diff --save_dir ./save/behave_enc_512 --dataset behave --save_interval 1000 --num_steps 20000 --arch trans_en
c --batch_size 24 --overwrite
```

Generate samples with HOI-DM:
```bash
python -m sample.local_generate_obj --model_path ./save/behave_enc_512/model000020000.pt --num_samples 10 --num_repetitions 1 --m
otion_length 10 --multi_backbone_split 4 --skip_first_stage
```


