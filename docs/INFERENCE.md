## Preparation
Before you start, please follow the instructions to prepare the dataset as described [here](https://github.com/Nicholasli1995/PCP-MV/blob/master/docs/DATASET.md). 

## Run a tracking-by-detection demo
Please follow this two-step guide:

(i) Run detection and save the results:
```bash
cd $PCP_MV_DIR
python -m torch.distributed.launch ./tools/test.py ./configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml $CKPT_PATH --eval bbox --out ./eval_results/nuscenes_demo.pkl
```
$CKPT_PATH is your model path where you can download a pre-trained demo model [here](https://drive.google.com/file/d/1N8PEmQtHjaSf12b4XD_XA8zf-7iVKazb/view?usp=sharing).

(ii) Run tracking based on the saved results:
 
```bash
cd $PCP_MV_DIR/tools/nusc_tracking/
python pub_test.py --checkpoint $PCP_MV_DIR/eval_results/results_nusc.json
```
You should see the results at $PCP_MV_DIR/tools/nusc_tracking/res/. This demo should give you a pedestrian tracking performance shown at the following table along with other references on the nuScenes validation set.

| Method                    | Reference|Overall AMOTA|Pedestrian AMOTA|
| ------------------------- | ---------------|  --------------| --------------| 
|[AB3DMOT](https://github.com/xinshuoweng/AB3DMOT)|IROS 2020|17.9| 9.1|
|[CenterPoint](https://github.com/tianweiy/CenterPoint)|CVPR 2021|65.9 |77.3 |
|[Probabilistic 3D MOT](https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking)|ICRA 2021|68.7| 76.6|
|[VoxelNeXt](https://github.com/dvlab-research/VoxelNeXt)|CVPR 2023|70.2 |N/A  |
|[JTD3D](https://github.com/TRAILab/jdt3d-website)|ECCV 2024|62.1 |N/A  |
|This demo           |ICRA 2025 |72.2|77.9|
