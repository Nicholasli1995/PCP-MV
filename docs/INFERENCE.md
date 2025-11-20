## Preparation
Before you start, please follow the instructions to prepare the dataset as described [here](https://github.com/Nicholasli1995/PCP-MV/blob/master/docs/DATASET.md). 

## Run a tracking-by-detection demo
Please follow this two-step guide:

(i) Modify the dataset_root parameter in the configuration file ($PCP_MV_DIR/configs/nuscenes/default.yaml) as your directory of the nuscenes dataset ($PCP_MV_DIR/data/nuscenes/). Run detection and save the results:
```bash
cd $PCP_MV_DIR
python -m torch.distributed.launch ./tools/test.py ./configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml $CKPT_PATH --eval bbox --out ./eval_results/nuscenes_demo.pkl
```
$CKPT_PATH is your model path where you can download pre-trained demo models as following:

| Model Name                                               | Use Density-aware weights | Use Relationship-aware Targets | Google Drive Link |
|----------------------------------------------------------|---------------------------|--------------------------------|-------------------|
| BEVFusion Baseline                                                 | ×                         | ×                              |[Link](https://drive.google.com/file/d/1N8PEmQtHjaSf12b4XD_XA8zf-7iVKazb/view?usp=sharing)                   |
| Baseline + the proposed representation learning approach | √                         | √                              |[Link](https://drive.google.com/file/d/1LMSyuMs2u6TXmb8YjF5EHoGZbAAdL85_/view?usp=sharing)                   |


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
|This demo (BEVFusion Baseline, NDS 71.35)![](https://github.com/Nicholasli1995/PCP-MV/blob/master/imgs/Baseline_Det_Eval.jpg)            |ICRA 2023 |72.6|77.7![](https://github.com/Nicholasli1995/PCP-MV/blob/master/imgs/Baseline_Tracking_Eval.jpg)|
|This demo (Baseline + the proposed representation learning approach, NDS 71.38)![](https://github.com/Nicholasli1995/PCP-MV/blob/master/imgs/Baseline%2BDW%2BRE_Det_Eval.jpg)          |ICRA 2025 |73.0|79.5![](https://github.com/Nicholasli1995/PCP-MV/blob/master/imgs/Baseline%2BDW%2BRE_Tracking_Eval.jpg)|

Notably, without incurring any extra inference cost, the proposed representation learning approach improves the performance for 3D object detection and tracking. 
