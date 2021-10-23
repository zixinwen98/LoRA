export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./adapter_lora_roberta_base_qqp"
python -m torch.distributed.launch --nproc_per_node=3 examples/text-classification/run_glue.py \
--model_name_or_path roberta-base \
--task_name qqp \
--do_train \
--do_eval \
--evaluation_strategy epoch \
--save_strategy epoch \
--max_seq_length 512 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--learning_rate 1e-3 \
--num_train_epochs 10 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--warmup_ratio 0.06 \
--apply_lora \
--lora_r 128 \
--lora_alpha 4 \
--apply_adapter \
--adapter_type houlsby \
--adapter_size 128 \
--seed 0 \
--weight_decay 0.1 \
--report_to all
