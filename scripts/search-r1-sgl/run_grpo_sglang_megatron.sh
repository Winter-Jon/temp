set -x
ray stop --force
HOME=/workspace/repository/verl

WAND_PROJECT="Qwen-14B-dev-test"
# WAND_PROJECT="dev-test"

export WANDB_API_KEY=fb5e61cc91c9c9bc4553bb55124bac77fda225f9
export RAY_DEDUP_LOGS=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# time=$(date +%Y-%m-%d-%H-%M-%S)

# comment="DCremoved_IS_cloned_traj_ratio_BHIS_4_dynamic_filter"
comment="Full_Tech_Prefix_Index"

# type="val_150steps"
type="train"

export BASE_MODEL='/workspace/modelsl/Qwen2.5-14B'
export EXPERIMENT_NAME=qwen2.5-14b-${comment}-${type}
# export EXPERIMENT_NAME=dev-test

search_r1_train_path=$HOME/data/search-r1-dataset/train.parquet
search_r1_test_path=$HOME/data/search-r1-dataset/test.parquet

train_files="['$search_r1_train_path']"
test_files="['$search_r1_test_path']"



python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.val_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.revision_prob=20 \
    data.filter_overlong_prompts=false \
    data.truncation='error' \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.fsdp_config.offload_policy=true \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24000 \
    actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_agent=5 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_max_token_len_per_gpu=98304 \
    critic.model.fsdp_config.param_offload=false \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=false \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=100 \
    trainer.total_epochs=30 \
    trainer.total_training_steps=1005 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    trainer.validation_data_dir=verl_checkpoints/$EXPERIMENT_NAME/val_generations \
    trainer.resume_mode=auto \
    reward_model.reward_manager=reward_manager \
    max_turns=5 \
    keep_ratio=0.12 \
    do_search=true \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
   "$@" 2>&1 | tee "$EXPERIMENT_NAME.log"


#     actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
#     critic.ppo_max_token_len_per_gpu=98304 \

#     actor_rollout_ref.actor.clip_ratio_high=0.28 \
#     actor_rollout_ref.actor.clip_ratio_low=0.2 \
