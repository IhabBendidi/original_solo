#!/bin/bash
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python3 main_pretrain.py --dataset cifar100  --no_labels \
    --backbone resnet50 \
    --data_dir datasets \
    --max_epochs 1000 \
    --devices 0 --accelerator gpu  --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 1.0 \
    --classifier_lr 0.1 \
    --weight_decay 1e-5 \
    --batch_size 512 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --nam byol${1} \
    --project Cifar_results \
    --entity labrats \
    --wandb \
    --offline \
    --save_checkpoint \
    --method byol \
    --proj_output_dim 256 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --momentum_classifier --color_jitter_prob ${1} --seed ${2}
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python3 main_linear.py \
    --dataset cifar100 \
    --backbone resnet50 \
    --data_dir datasets \
    --train_dir cifar100/train \
    --val_dir cifar100/val \
    --max_epochs 100 \
    --devices 0 --accelerator gpu  --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 4 \
    --nam byol${1}  \
    --pretrained_feature_extractor lorepm_ipsum.ckpt \
    --project Cifar_results \
    --entity labrats \
    --wandb --offline