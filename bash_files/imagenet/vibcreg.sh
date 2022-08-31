#!/bin/bash
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python3 main_pretrain.py --dataset imagenet --no_labels  \
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
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --num_workers 4 \
    --min_scale 0.2 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --num_crops_per_aug 1 1 \
    --nam vibcreg${1} \
    --project Cifar_results \
    --entity labrats \
    --wandb \
    --offline \
    --save_checkpoint \
    --method vibcreg \
    --proj_hidden_dim 2048 \
    --proj_output_dim 2048 \
    --sim_loss_weight 25.0 \
    --var_loss_weight 25.0 \
    --cov_loss_weight 200.0 \
    --iternorm --color_jitter_prob ${1}  --dali --seed ${2}
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
    --nam vibcreg${1}  \
    --pretrained_feature_extractor lorepm_ipsum.ckpt \
    --project Cifar_results \
    --entity labrats \
    --wandb --offline --dali