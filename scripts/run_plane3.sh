#! /bin/bash
# train 1m command
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python train_nerf.py data/nerf_synthetic/lego --workspace trial_plane3_1m --fp16 --plane3 --cuda_ray --bound 1 --scale 0.8 --mode blender  --num_epoch 10000 --eval_interval 100