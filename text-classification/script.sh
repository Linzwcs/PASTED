
plm_dir="allenai/longformer-base-4096"
seed=42629309
dataset_name="linzw/PASTED"
task_name="text-classification"
out_dir="../model/${task_name}_${seed}"
time=$(date +'%m:%d:%H:%M')
mkdir -p $out_dir

CUDA_VISIBLE_DEVICES=0 nohup python3 main.py \
  --do_train \
  --do_eval \
  --do_predict \
  --model_name_or_path $plm_dir \
  --dataset_name $dataset_name\
  --task_name $task_name \
  --dataloader_num_workers 2 \
  --dataloader_prefetch_factor 2 \
  --max_seq_length 2048 \
  --per_device_train_batch_size  12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --evaluation_strategy steps \
  --eval_steps 1500 \
  --overwrite_output_dir \
  --save_total_limit 2 \
  --fp16 \
  --max_train_samples 100 \
  --output_dir $out_dir 2>&1 | tee $out_dir/log.train.$time   

