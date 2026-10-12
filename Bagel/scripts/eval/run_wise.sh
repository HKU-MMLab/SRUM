#!/bin/bash
set -x

PYTHON=/pfs/weiyang/Miniconda3/envs/bagel/bin

# echo "⏳ 等待 2 小时..."
sleep 1

GPUS=8
cd /pfs/weiyang/WISE_re/Bagel
export PYTHONPATH=/pfs/weiyang/WISE_re/Bagel:$PYTHONPATH

echo "🚀 第一段"

$PYTHON/torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12347 \
    eval/gen/gen_images_mp_wise.py \
    --output_dir /pfs/weiyang/WISE_re/WISE_Verified_Result/Bagel \
    --metadata_file /pfs/weiyang/WISE_re/WISE/data_verified/merge_v2.json \
    --resolution 1024 \
    --max_latent_size 64 \
    --model-path /pfs/weiyang/WISE_re/CKPT/ByteDance-Seed/BAGEL-7B-MoT

echo "🚀 第二段"

$PYTHON/torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12349 \
    eval/gen/gen_images_mp_wise.py \
    --output_dir /pfs/weiyang/WISE_re/WISE_Verified_Result/Bagel-think \
    --metadata_file /pfs/weiyang/WISE_re/WISE/data_verified/merge_v2.json \
    --resolution 1024 \
    --max_latent_size 64 \
    --model-path /pfs/weiyang/WISE_re/CKPT/ByteDance-Seed/BAGEL-7B-MoT \
    --think