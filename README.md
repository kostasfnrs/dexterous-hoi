# Language-conditioned HOI generation using the GRAB dataset

Activate the environment:

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

```