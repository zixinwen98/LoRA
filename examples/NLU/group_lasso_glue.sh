export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./trial_run_group_lasso_roberta_base_mnli"

for gl_param in 10
do
for lr in 1e-5
do
python examples/group-lasso-text-classification/run_glue.py \
--model_name_or_path roberta-base \
--task_name mnli \
--do_train \
--do_eval \
--evaluation_strategy epoch \
--save_strategy epoch \
--max_seq_length 128 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--learning_rate $lr \
--num_train_epochs 1 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--warmup_ratio 0.06 \
--seed 0 \
--report_to all \
--glasso_param $gl_param
done
done