export GPU_NUM=1
# A baseline configuration
export CFG_PATH='./configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml'
export RUN_DIR='./training_exps/debug_train/'

# Local run with single GPU
python -m torch.distributed.launch \
--nproc_per_node=$GPU_NUM \
--nnodes=1 \
--node_rank=0 \
--master_addr=localhost \
--master_port=8088 \
./tools/train.py \
$CFG_PATH \
--run-dir $RUN_DIR \
--model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
--load_from \
pretrained/lidar-only-det.pth 

# Distributed training (you need to specify the environment variables)
# python -m torch.distributed.launch \
# --nproc_per_node=$GPU_NUM \
# --nnodes=$N_NODES \
# --node_rank=$NODE_RANK \
# --master_addr=$MASTER_ADDR \
# --master_port=$MASTER_PORT \
# ./tools/train.py \
# $CFG_PATH \
# --run-dir $RUN_DIR \
# --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
# --load_from \
# pretrained/lidar-only-det.pth 
