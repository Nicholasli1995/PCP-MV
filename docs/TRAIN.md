
## Training on the nuScenes dataset
Before you start, please follow the instructions to prepare the dataset as described [here](https://github.com/Nicholasli1995/PCP-MV/blob/master/docs/DATASET.md).

You can download some pre-trained weights by running
```bash
cd $PCP_MV_DIR
bash ./tools/download_pretrained.sh
```

You can then try training a baseline model with the script:
```bash
cd $PCP_MV_DIR
bash ./scripts/train.sh
```
This script defaults to local training and you can adjust it for multi-node distributed training. Various training configurations are being added.

## Training using the configurations of PCP-MV


