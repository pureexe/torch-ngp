#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_nerf.py data/nerf_synthetic/lego --workspace trial_fibonacci_test --fp16 --bound 1 --scale 0.8 --mode blender --fibonacci 30 --num_steps 64 --max_ray_batch 4096 --global_tri --blend_lattice 2 
