#!/bin/bash

# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE=1

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="train_ddp.py"

# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================

USE_DISTILL_LOSS=${1:-True}

USE_SIGREG_LOSS=${2:-True}

KD_WEIGHT=${3:-10}

SIGREG_WEIGHT=${4:-0.1}

NUM_T=${5:-17}

T_MAX=${6:-5}

NUM_LAYER=${7:-24}

PORT=${8:-29511}

# ============================================================
# Convert True/False -> 1/0 cho tên folder
# ============================================================

bool_to_python() {
    case "$1" in
        1|true|True|TRUE)
            echo "True"
            ;;
        0|false|False|FALSE)
            echo "False"
            ;;
        *)
            echo "ERROR: Boolean argument must be 0/1 or True/False, got '$1'" >&2
            exit 1
            ;;
    esac
}

bool_to_int() {
    case "$1" in
        1|true|True|TRUE)
            echo "1"
            ;;
        0|false|False|FALSE)
            echo "0"
            ;;
        *)
            echo "ERROR: Boolean argument must be 0/1 or True/False, got '$1'" >&2
            exit 1
            ;;
    esac
}


# Python values
DISTILL_LOSS_BOOL=$(bool_to_python "$USE_DISTILL_LOSS")
SIGREG_BOOL=$(bool_to_python "$USE_SIGREG_LOSS")

# Folder values
D_DISTILL=$(bool_to_int "$USE_DISTILL_LOSS")
D_SIGREG=$(bool_to_int "$USE_SIGREG_LOSS")

# ============================================================
# Tên experiment
# ============================================================

# EXP_NAME="cosine_jepa_d${D_DISTILL}_cse${D_CSE}_vis${D_VISION}_sig${D_SIGREG}_kd${KD_WEIGHT}_sw${SIGREG_WEIGHT}_l${NUM_LAYER}_dt${D_TAU}"
# EXP_NAME="struct_sigreg_JL_gmm_d${D_DISTILL}_sig${D_SIGREG}_kd${KD_WEIGHT}_sw${SIGREG_WEIGHT}_numt${NUM_T}_tmax${T_MAX}_l${NUM_LAYER}"
EXP_NAME="struct_sigreg_SW_gmm_d${D_DISTILL}_sig${D_SIGREG}_kd${KD_WEIGHT}_sw${SIGREG_WEIGHT}_numt${NUM_T}_tmax${T_MAX}_l${NUM_LAYER}"

OUTPUT_DIR="training/FastVLM-0.5B_vqa_${EXP_NAME}"
CACHE_DIR="caching/B3_Qwen2_2B_vqa"

echo "============================================================"
echo "Experiment:"
echo "  USE_DISTILL_LOSS      = $USE_DISTILL_LOSS"
echo "  USE_SIGREG_LOSS       = $USE_SIGREG_LOSS"
echo "  KD_WEIGHT             = $KD_WEIGHT"
echo "  SIGREG_WEIGHT         = $SIGREG_WEIGHT"
echo "  NUM_LAYER             = $NUM_LAYER"
echo "  NUM_T                 = $NUM_T"
echo "  T_MAX                 = $T_MAX"
echo ""
echo "OUTPUT_DIR:"
echo "  $OUTPUT_DIR"
echo "============================================================"

torchrun  \
    --master_addr=127.0.0.1 --master_port=$PORT \
    --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
    --model_name apple/FastVLM-0.5B \
    --lora True \
    --teacher_lora True \
    --lora_r 64 \
    --lora_alpha 64 \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --percent_data 1.0 \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --bf16 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --seed 42 \
    --weight_decay 0.01 \
    --normalize True \
    --lr_scheduler_type "constant" \
    --warmup_ratio 0.05 \
    --caching_dir "$CACHE_DIR" \
    --kd_loss_type "daved" \
    --image_resolution "low" \
    --projector_config_path "./config/projector_config_emo.json" \
    --projector_lr 5e-5 \
    --report_to None \
    --gmm_ckpt "gmm_training/B3_Qwen2_2B_vqa/gmm.joblib" \
    --use_distill_loss "$DISTILL_LOSS_BOOL" \
    --use_sigreg_loss "$SIGREG_BOOL" \
    --kd_weight "$KD_WEIGHT" \
    --sigreg_weight "$SIGREG_WEIGHT" \
    --num_t "$NUM_T" \
    --t_max "$T_MAX" \
    --num_layers "$NUM_LAYER" 


# ============================================================
# 2. CHECKPOINT
# ============================================================

MODEL="$OUTPUT_DIR/checkpoint-epoch-0"

if [ ! -d "$MODEL" ]; then
    echo "ERROR: Checkpoint not found:"
    echo "$MODEL"
    exit 1
fi

echo ""
echo "============================================================"
echo "Training finished."
echo "Checkpoint:"
echo "  $MODEL"
echo "============================================================"


SUBSETS=(
  "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W"
  "ScienceQA" "VizWiz" "GQA" "TextVQA"
)

EVAL_OUTPUT="./MMEB-eval_outputs/FastVLM-0.5B_vqa_${EXP_NAME}/"

python eval_mmeb.py \
    --model_name "$MODEL" \
    --encode_output_path "$EVAL_OUTPUT" \
    --lora True --lora_r 64 --lora_alpha 64 \
    --pooling eos \
    --model_backbone llava_qwen2 \
    --normalize True \
    --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split test \
    --per_device_eval_batch_size 32 \
    --image_dir eval_images/ \
    --tgt_prefix_mod \
    --load_pretrained_lora True \
    --report_to none


echo ""
echo "============================================================"
echo "DONE"
echo "Experiment:"
echo "  $EXP_NAME"
echo ""
echo "Train:"
echo "  $OUTPUT_DIR"
echo ""
echo "Eval:"
echo "  $EVAL_OUTPUT"
echo "============================================================"