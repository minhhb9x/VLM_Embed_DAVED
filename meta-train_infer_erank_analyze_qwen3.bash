#!/bin/bash

INFER_SCRIPT="infer_eval_hidden_attention_qwen3.py"   # sửa thành script infer của bạn

INFER_SUBSETS=(
    "ImageNet-1K"
    # "N24News" "HatefulMemes" "VOC2007" "SUN397"
    # "Place365" "ImageNet-A" "ImageNet-R" "ObjectNet" "Country211"
)

MODELS=(
    "Qwen/Qwen3-VL-Embedding-8B"
)

BACKBONES=(
    "qwen3_vl"
)

# Kiểm tra MODELS và BACKBONES có cùng số phần tử không
if [ "${#MODELS[@]}" -ne "${#BACKBONES[@]}" ]; then
    echo "Error: MODELS and BACKBONES must have the same length."
    exit 1
fi

mkdir -p infer
mkdir -p analyze

for i in "${!MODELS[@]}"; do

    MODEL="${MODELS[$i]}"
    BACKBONE="${BACKBONES[$i]}"

    EXP_NAME=$(basename "$(dirname "$MODEL")")

    EXTRA_ARGS=()
    OUTPUT_SUFFIX=""

    if [ "$BACKBONE" = "llava_onevision" ] || [ "$BACKBONE" = "llava_onevision_old" ]; then
        IMAGE_RESOLUTION="tiny"

        EXTRA_ARGS+=(--image_resolution "$IMAGE_RESOLUTION")
        OUTPUT_SUFFIX="_${IMAGE_RESOLUTION}"
    fi

    INFER_OUTPUT="infer/${EXP_NAME}${OUTPUT_SUFFIX}"
    ANALYZE_OUTPUT="analyze/${EXP_NAME}${OUTPUT_SUFFIX}.txt"

    echo "========================================================"
    echo "Model     : $MODEL"
    echo "Backbone  : $BACKBONE"
    echo "Experiment: $EXP_NAME"
    echo "Infer out : $INFER_OUTPUT"
    echo "Analyze   : $ANALYZE_OUTPUT"
    echo "Extra args: ${EXTRA_ARGS[*]}"
    echo "========================================================"

    CUDA_VISIBLE_DEVICES=0 python "$INFER_SCRIPT" \
        --model_name "$MODEL" \
        --pooling eos \
        --model_backbone "$BACKBONE" \
        --normalize True \
        --bf16 \
        --dataset_name "TIGER-Lab/MMEB-eval" \
        --subset_name "${INFER_SUBSETS[@]}" \
        --dataset_split "test" \
        --image_dir "eval_images/" \
        --tgt_prefix_mod \
        --encode_output_path "$INFER_OUTPUT" \
        --per_device_eval_batch_size 1 \
        --report_to None \
        "${EXTRA_ARGS[@]}"

    if [ $? -ne 0 ]; then
        echo "ERROR: Infer failed for $EXP_NAME"
        continue
    fi

    CUDA_VISIBLE_DEVICES=0 python er_statistic_qwen3.py \
        --pt_dir "${INFER_OUTPUT}/${INFER_SUBSETS[0]}/query" \
        --num_samples 0 \
        --normalize_by_min_dim \
        --output_file "${ANALYZE_OUTPUT}"
    
    # rm -rf "${INFER_OUTPUT}"

    echo "Finished: ${EXP_NAME}${OUTPUT_SUFFIX}"
    echo

done