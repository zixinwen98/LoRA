export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./adapter_lora_roberta_base_qqp"

for gl_param in 0.01 0.05 0.1 0.3
do
python -m torch.distributed.launch --nproc_per_node=8 \
    examples/group-lasso-text-classification/run_glue.py \
--model_name_or_path roberta-base \
--task_name mnli \
--do_train \
--do_eval \
--evaluation_strategy epoch \
--save_strategy epoch \
--max_seq_length 128 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--learning_rate 1e-2 \
--num_train_epochs 10 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--warmup_ratio 0.06 \
--seed 0 \
--report_to all \
--glasso_param $gl_param
done