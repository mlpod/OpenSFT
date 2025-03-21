set -x
export HOME=/tmp
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:2048'

ip=`hostname -i`

project_name=Qwen2.5-72B-Instruct
model_path=path_to_model
data_path=../data/dataset/distill_r1_110k.packed.jsonl
data_config_path=../data/config/distill_r1_110k.json
save_checkpoint=path_to_save_dir

python config.py --max_steps=2000 --learning_rate=2e-5 --warmup_num_steps=200

master_ip=192.168.1.1 # 主节点ip
num_machines=24
num_processes=192
machine_rank=0 # 0到num_machines-1，每台机器一个rank

nohup accelerate launch \
    --config_file config/multi_node.yaml \
    --num_processes=$num_processes \
    --num_machines=$num_machines \
    --machine_rank=$machine_rank \
    --main_process_ip=$master_ip \
    train.py \
    --project_name $project_name \
    --gradient_accumulation_steps 1 \
    --save_checkpoint $save_checkpoint \
    --seed 2025 \
    --model_path $model_path \
    --data_path $data_path \
    --data_config_path $data_config_path \
    --sequence_parallel_degree 8 \
    --num_epochs 4 > logs/$ip.log 2>&1 &