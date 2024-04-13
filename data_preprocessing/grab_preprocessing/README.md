# Convert GRAB dataset to BEHAVE format

This is the first preprocessing step in the pipeline: convert GRAB to BEHAVE format and then use the converted dataset with the converter from BEHAVE to HumanML3d.

First, download the zip files from GRAB and put them in a directory (in 14:30:27 following, this will be referred to as grap_zips). Then, run

```python
python grab/unzip_grab.py --grab-path /media/erik/DATA/grab_zips --
extract-path /media/erik/DATA/grab_extracted
```

where you adjust the paths to your liking.

Download the GRAB data from AMASS. Download the MANO models and put them in `./data_preprocessing/grab_preprocessing/body_models/mano/`. 

Then, in `converter.ipynb` adjust the paths according to your wishes/download locations and execute the notebook cells.




