data_name=search-r1-dataset

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR=data/${data_name} # first download the data from https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train

export PYTHONPATH=/data/home/zhangjing1/repos/fengxiang/rag_reasoning/rag-reasoning-explicit/Megatron-LM:$PYTHONPATH

WAND_PROJECT="RAG-Reasoning-Explicit"


# export CUDA_LAUNCH_BLOCKING=1

export WANDB_API_KEY=fb5e61cc91c9c9bc4553bb55124bac77fda225f9

# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export CUDA_LAUNCH_BLOCKING=1   # 调试专用，真机训练按需关掉
# export NCCL_BLOCKING_WAIT=1 
# export TORCH_USE_CUDA_DSA=1

export TRAIN_DATA_DIR=/data/home/zhangjing1/repos/fengxiang/rag_reasoning/rag-reasoning-explicit/data/$data_name
export TEST_DATA_DIR=/data/home/zhangjing1/repos/fengxiang/rag_reasoning/rag-reasoning-explicit/data/$data_name

train_data=search-r1-dataset
test_data=search-r1-dataset

time=$(date +%Y-%m-%d-%H-%M-%S)

# export BASE_MODEL='Qwen/Qwen2.5-3B'
# export EXPERIMENT_NAME=${train_data}-${test_data}-search-r1-grpo-qwen2.5-3b-em
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=${train_data}-${test_data}-search-r1-grpo-qwen2.5-3b-it-em
export BASE_MODEL='/data/home/zhangjing1/repos/anesbench/share/models/Qwen2.5-3B'
export EXPERIMENT_NAME=grpo-qwen2.5-3b-em-${time}-${train_data}-${test_data}
# export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=${train_data}-${test_data}-search-r1-grpo-qwen2.5-7b-it-em
# export BASE_MODEL='Qwen/Qwen2.5-14B'
# export EXPERIMENT_NAME=${train_data}-${test_data}-search-r1-grpo-qwen2.5-14b-em
# export BASE_MODEL='Qwen/Qwen2.5-14B-Instruct'
# export EXPERIMENT_NAME=${train_data}-${test_data}-search-r1-grpo-qwen2.5-14b-it-em

export RAY_PROFILING=1                       # 打开事件采样
export RAY_task_events_report_interval_ms=0  # 采样全部事件

# set -x
ray stop --force
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues
# ray start --head --num-cpus=32 --num-gpus=8
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_DATA_DIR/train.parquet \
    data.val_files=$TEST_DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.max_revision_times=1 \
    data.rollout_times=5 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb','console'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=100 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=30 \
    trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=5 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log


    # actor_rollout_ref.actor.strategy=megatron \
    # critic.strategy=megatron \