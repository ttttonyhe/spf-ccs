CUDA_VISIBLE_DEVICES=4,5,6 accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
  --num_processes 3 \
  --main_process_port 29501 \
  finetune.py --model_name_or_path='Qwen/Qwen2.5-7B-Instruct' \
  --dataset_name='pure_bad' --model_family='qwen2' --learning_rate=2e-5 \
  --per_device_train_batch_size=16 --gradient_accumulation_steps=1 \
  --output_dir='outputs/pure_bad/qwen_25_7b_spf' \
  --logging_steps=1 --num_train_epochs=5 --gradient_checkpointing --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' \
  --sft_type='sft' \
  --use_warmup=True \
  --use_spf=True


# CUDA_VISIBLE_DEVICES=6 accelerate launch --num_processes=1 \
# 	eval_safety.py \
# 	--torch_dtype=bfloat16 --model_name_or_path="outputs/pure_bad/qwen_25_7b_spf" \
# 	--safety_bench='hex-phi' --model_family='qwen2' \
#   	--prompt_style='qwen2' --evaluator='harmbench' \
#   	--save_path='results/asr/qwen_25_7b_pure_bad.json' --eval_template='pure_bad';