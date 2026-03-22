#!/bin/bash
set -euo pipefail

USE_SPF=False

TRAIN_CUDA_DEVICES="0,1,2,3,4,5"
EVAL_CUDA_DEVICES="6"
TRAIN_NUM_PROCESSES=6
EVAL_NUM_PROCESSES=1
MODEL_NAME_OR_PATH="meta-llama/Llama-3.1-8B-Instruct"
MODEL_FAMILY="llama3"
DATASET_NAME="gsm8k"
LEARNING_RATE="2e-5"
TRAIN_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=1
OUTPUT_DIR="outputs/math/llama_sft"
RESULTS_PATH="results/util/math_llama_sft.json"

CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_DEVICES}" accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
  --num_processes "${TRAIN_NUM_PROCESSES}" \
  finetune.py --model_name_or_path="${MODEL_NAME_OR_PATH}" \
  --dataset_name="${DATASET_NAME}" --model_family="${MODEL_FAMILY}" --learning_rate="${LEARNING_RATE}" \
  --per_device_train_batch_size="${TRAIN_BATCH_SIZE}" --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
  --output_dir="${OUTPUT_DIR}" \
  --logging_steps=1 --num_train_epochs=5 --gradient_checkpointing --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' \
  --sft_type='sft' \
  --use_warmup=True \
  --use_spf="${USE_SPF}"

CUDA_VISIBLE_DEVICES="${EVAL_CUDA_DEVICES}" accelerate launch --num_processes="${EVAL_NUM_PROCESSES}" \
  eval_utility.py \
  --torch_dtype=bfloat16 \
  --model_name_or_path="${OUTPUT_DIR}" \
  --dataset="${DATASET_NAME}" \
  --model_family="${MODEL_FAMILY}" \
  --prompt_style="${MODEL_FAMILY}" \
  --evaluator='rouge_1' \
  --save_path="${RESULTS_PATH}"