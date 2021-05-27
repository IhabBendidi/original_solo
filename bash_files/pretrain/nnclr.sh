python3 ../../main_contrastive.py \
    --dataset imagenet100 \
    --encoder resnet18 \
    --data_folder /datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/val \
    --epochs 500 \
    --optimizer sgd \
    --lars \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.4 \
    --weight_decay 1e-5 \
    --batch_size 128 \
    --gpus 0 1 \
    --num_workers 10 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --asymmetric_augmentations \
    --name nnclr-0.2-500ep \
    --project debug \
    --wandb \
    nnclr \
    --temperature 0.2 \
    --hidden_dim 2048 \
    --pred_hidden_dim 4096 \
    --encoding_dim 256 \
    --queue_size 65536