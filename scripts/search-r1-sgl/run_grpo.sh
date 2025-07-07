set -x
ray stop --force
HOME=/workspace/repository/verl

WAND_PROJECT="dev-test"

export WANDB_API_KEY=fb5e61cc91c9c9bc4553bb55124bac77fda225f9
export RAY_DEDUP_LOGS=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
# 加到 ray 的启动变量里

# gsm8k_train_path=$HOME/data/gsm8k/train.parquet
# gsm8k_test_path=$HOME/data/gsm8k/test.parquet
# math_train_path=$HOME/data/math/train.parquet
# math_test_path=$HOME/data/math/test.parquet

export BASE_MODEL='/workspace/modelsl/Qwen2.5-3B'
export EXPERIMENT_NAME=grpo-qwen2.5-3b-em-${time}-${train_data}-${test_data}

search_r1_train_path=$HOME/data/search-r1-dataset/train.parquet
search_r1_test_path=$HOME/data/search-r1-dataset/test.parquet

train_files="['$search_r1_train_path']"
test_files="['$search_r1_test_path']"

python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.revision_prob=0.2 \
    data.max_response_length=500 \
    data.filter_overlong_prompts=false \
    data.truncation='error' \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    algorithm.use_kl_in_reward=false \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=100 \
    trainer.val_before_train=false \
    trainer.total_epochs=30 \
    trainer.total_training_steps=1005 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    reward_model.reward_manager=reward_manager \
    max_turns=5 \
    do_search=true \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
   "$@" 2>&1 | tee "$EXPERIMENT_NAME.log"