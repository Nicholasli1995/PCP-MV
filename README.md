# PCP-MV

Official project website for the ICRA 2025 paper "[Learning better representations for crowded pedestrians in offboard LiDAR-camera 3D tracking-by-detection](https://arxiv.org/abs/2505.16029)". 

[[Poster](https://drive.google.com/file/d/1Q3g4COChyJZ3smdSxP5NKc5Mt1PWNPk_/view?usp=sharing)]

## Environment
Before you start, please refer to [ENV.md](https://github.com/Nicholasli1995/PCP-MV/blob/master/docs/ENV.md) to build this project.

## Data preparation
Please follow [DATASET.md](https://github.com/Nicholasli1995/PCP-MV/blob/master/docs/DATASET.md) to download and prepare the nuScenes dataset.

## Usage: inference demo
Refer to [INFERENCE.md](https://github.com/Nicholasli1995/PCP-MV/blob/master/docs/INFERENCE.md) to perform LiDAR-camera tracking-by-detection and quantitative evaluation.

## Usage: training experiments
Refer to [TRAIN.md](https://github.com/Nicholasli1995/PCP-MV/blob/master/docs/TRAIN.md) to perform training with various configurations.

## License
A MIT license is used for this repository. Third-party datasets like nuScenes are subject to their own licenses and the user should obey them strictly.

## Acknowledgement
This repository is developed based on [BEVFusion](https://github.com/mit-han-lab/bevfusion). Thank the authors for their contributions.

## Citation
Please star this repository and cite the following paper in your publications if it helps your research:

    @INPROCEEDINGS{11128508,
      author={Li, Shichao and Li, Peiliang and Lian, Qing and Yun, Peng and Chen, Xiaozhi},
      booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)}, 
      title={Learning Better Representations for Crowded Pedestrians in Offboard LiDAR-Camera 3D Tracking-by-detection}, 
      year={2025},
      volume={},
      number={},
      pages={2740-2747},
      keywords={Point cloud compression;Training;Pedestrians;Three-dimensional displays;Urban areas;Benchmark testing;Semisupervised learning;Trajectory;Robotics and automation;System analysis and design},
      doi={10.1109/ICRA55743.2025.11128508}
      }

[Link to the paper](https://arxiv.org/abs/2505.16029)
