## Dataset Preparation: nuScenes

For experiments on the nuScenes dataset, please finish the following steps:

(i). Download the nuScenes 3D detection dataset [HERE](https://www.nuscenes.org/download) and unzip all .zip files at your dataset directory $NUSCENES_DATA_DIR.

(ii). Create a symbolic link to the nuScenes dataset:
```bash
ln -s $NUSCENES_DATA_DIR ./data/nuscenes
```

Your folder structure should be organized as follows:

```
PCP-MV
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```

(iii) Run the following command to prepare the datalists used for training and evaluation

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

After this step, you should see the following directory structure with the prepared .pkl files :

```
PCP-MV
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl

```

If you do not want to run the previous command, you can download the prepared .pkl files here (To be updated).