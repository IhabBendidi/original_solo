#!/bin/bash
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python3 main_pretrain.py --dataset imagenet --no_labels \
    --backbone resnet50 \
    --data_dir datasets \
    --train_dir imagenet/train \
    --val_dir imagenet/val \
    --max_epochs 100 \
    --devices 0 --accelerator gpu  --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --scheduler warmup_cosine \
    --lr 0.6 \
    --min_lr 0.0006 \
    --classifier_lr 0.1 \
    --weight_decay 1e-6 \
    --batch_size 128 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --num_crops_per_aug 1 1 \
    --save_checkpoint \
    --nam swav${1} \
    --project Cifar_results \
    --entity labrats \
    --wandb \
    --offline \
    --method swav \
    --proj_hidden_dim 2048 \
    --queue_size 3840 \
    --proj_output_dim 128 \
    --num_prototypes 3000 \
    --epoch_queue_starts 50 \
    --freeze_prototypes_epochs 2 --color_jitter_prob ${1} --dali
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python3 main_linear.py \
    --dataset imagenet \
    --backbone resnet50 \
    --data_dir datasets \
    --train_dir imagenet/train \
    --val_dir imagenet/val \
    --max_epochs 100 \
    --devices 0 --accelerator gpu  --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 128 \
    --num_workers 4 \
    --nam swav${1}  \
    --pretrained_feature_extractor lorepm_ipsum.ckpt \
    --project Cifar_results \
    --entity labrats \
    --wandb --offline --dali