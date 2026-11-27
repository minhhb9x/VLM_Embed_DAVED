#!/usr/bin/env bash
# Train FastVLM on 5 CLS datasets, then evaluate all 10 IND/OOD CLS datasets.
# Usage: bash scripts/train_distill_coral_cls.sh [distribution_weight]
set -euo pipefail

WEIGHT=${1:-0.1}
OUTPUT_DIR="training/FastVLM-0.5B_cls_coral_w${WEIGHT}"

torchrun --standalone --nproc_per_node=1 train_ddp.py \
    --model_name apple/FastVLM-0.5B \
    --model_backbone llava_qwen2 \
    --student_hidden_dim 896 --teacher_hidden_dim 1536 \
    --lora True --lora_r 64 --lora_alpha 64 \
    --pooling eos --normalize True \
    --dataset_name TIGER-Lab/MMEB-train \
    --dataset_split original \
    --subset_name ImageNet_1K N24News HatefulMemes VOC2007 SUN397 \
    --image_dir vlm2vec_train/MMEB-train \
    --image_resolution low --percent_data 1.0 \
    --caching_dir caching/B3_Qwen2_2B_cls \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --projector_config_path config/projector_config_distribution.json \
    --projector_lr 5e-5 \
    --num_train_epochs 1 --bf16 \
    --weight_decay 0.01 \
    --lr_scheduler_type constant --warmup_ratio 0.05 \
    --save_strategy epoch --save_total_limit 5 --logging_steps 1 \
    --seed 42 --report_to none \
    --kd_loss_type coral_kd \
    --use_distill_loss True --kd_weight 10 \
    --use_sigreg_loss True --distribution_weight "$WEIGHT" \
    --num_layers 24 \
    --distribution_normalize False

MODEL="$OUTPUT_DIR/checkpoint-epoch-0"
if [[ ! -d "$MODEL" ]]; then
    echo "Checkpoint not found: $MODEL" >&2
    exit 1
fi

python eval_mmeb.py \
    --model_name "$MODEL" \
    --encode_output_path "MMEB-eval_outputs/FastVLM-0.5B_cls_coral_w${WEIGHT}" \
    --lora True --lora_r 64 --lora_alpha 64 \
    --pooling eos --model_backbone llava_qwen2 \
    --normalize True --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name ImageNet-1K N24News HatefulMemes VOC2007 SUN397 \
        Place365 ImageNet-A ImageNet-R ObjectNet Country211 \
    --dataset_split test \
    --per_device_eval_batch_size 32 \
    --image_dir eval_images/ \
    --tgt_prefix_mod \
    --load_pretrained_lora True \
    --report_to none
