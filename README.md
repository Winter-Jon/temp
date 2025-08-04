# REX-RAG
The officail implement of REX-RAG

### Core Algo:
- Rollout Phase: search_r1/llm_agent/generation.py
  - Exploration Prompts: search_r1/llm_agent/thinking_prompts.py
- Update Phase: verl/workers/actor/dp_actor.py
  - Trajectory Filter: verl/trainer/ppo/ray_trainer.py 1534-1547
  - Probe Policy Definition: verl/workers/actor/exploration_distribution.py
  - Importance Sampling: verl/workers/actor/dp_actor.py 448-498


### How to run:
1. Prepare Environment following verl
2. Prepare retriever, run search_r1/search/retrieval_server.py
3. Run scripts/search-r1-sgl/run_ppo_sglang_fsdp.sh