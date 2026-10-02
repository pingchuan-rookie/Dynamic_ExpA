# Training parameters and metrics

Last updated: 07/02/2026.

See [advanced NPU features](../../feature_support/npu_advance_features.md).

verl uses hierarchical YAML under verl/trainer/config. These upstream defaults are a reference; Dynamic-ExpA public experiment configuration determines allowed datasets and budgets.

---

## 1. Configuration

### 1.1 Shared parameters

These parameters have shared meanings in FSDP and Megatron.

#### 1.1.1 Actor optimizer

| Parameter | Default | Description |
|--------|--------|------|
| `actor_rollout_ref.actor.optim.lr` | `1.0e-06` | Actor learning rate |
| `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio` | `0.0` | Warmup steps as a fraction of total training steps |
| `actor_rollout_ref.actor.optim.total_training_steps` | `-1` | Total steps; -1 computes automatically |
| `actor_rollout_ref.actor.optim.weight_decay` | `0.01` | Weight decay |
| `actor_rollout_ref.actor.optim.lr_warmup_steps` | `-1` | Warmup steps; -1 derives from the ratio |
| `actor_rollout_ref.actor.optim.betas` | `[0.9, 0.999]` | Adam first/second moment coefficients |
| `actor_rollout_ref.actor.optim.clip_grad` | `1.0` | Gradient clipping threshold |
| `actor_rollout_ref.actor.optim.override_optimizer_config` | `null` / `{}` | Optimizer overrides: null for FSDP, {} for Megatron |

#### 1.1.2 Actor policy

| Parameter | Default | Description |
|--------|--------|------|
| `actor_rollout_ref.actor.strategy` | `fsdp` / `megatron` | Training strategy: fsdp or megatron |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | `256` | PPO mini-batch size |
| `actor_rollout_ref.actor.ppo_micro_batch_size` | `null` | PPO micro-batch size |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | `null` | PPO micro-batch size per GPU |
| `actor_rollout_ref.actor.use_dynamic_bsz` | `false` | Dynamic batching |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | `16384` | PPO token limit per GPU |
| `actor_rollout_ref.actor.clip_ratio` | `0.2` | PPO clipping ratio; commonly 0.1-0.3 |
| `actor_rollout_ref.actor.clip_ratio_low` | `0.2` | PPO lower clipping ratio |
| `actor_rollout_ref.actor.clip_ratio_high` | `0.2` | PPO upper clipping ratio |
| `actor_rollout_ref.actor.tau_pos` | `1.0` | Tau for positive advantages |
| `actor_rollout_ref.actor.tau_neg` | `1.05` | Tau for negative advantages |
| `actor_rollout_ref.actor.freeze_vision_tower` | `false` | Freeze the multimodal vision tower |
| `actor_rollout_ref.actor.clip_ratio_c` | `3.0` | Dual-clip upper constant |
| `actor_rollout_ref.actor.loss_agg_mode` | `token-mean` | Loss aggregation, such as token-mean |
| `actor_rollout_ref.actor.loss_scale_factor` | `null` | Loss scaling factor |
| `actor_rollout_ref.actor.entropy_coeff` | `0` | Entropy regularization coefficient |
| `actor_rollout_ref.actor.calculate_entropy` | `false` | Compute policy entropy |
| `actor_rollout_ref.actor.use_kl_loss` | `false` | Enable KL loss |
| `actor_rollout_ref.actor.use_prefix_grouper` | `false` | Enable prefix grouping |
| `actor_rollout_ref.actor.use_torch_compile` | `true` | Enable torch.compile |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.001` | KL loss coefficient |
| `actor_rollout_ref.actor.kl_loss_type` | `low_var_kl` | KL estimator, such as low_var_kl |
| `actor_rollout_ref.actor.ppo_epochs` | `1` | PPO update epochs |
| `actor_rollout_ref.actor.shuffle` | `false` | Shuffle mini-batches |
| `actor_rollout_ref.actor.data_loader_seed` | `42` | Data-loader seed |
| `actor_rollout_ref.actor.grad_clip` | `1.0` | Gradient clipping threshold |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size` | `1` | Ulysses parallel degree |
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking` | `false` | Chunk entropy computation from logits |
| `actor_rollout_ref.actor.entropy_from_logits_chunk_size` | `2048` | Entropy chunk size |
| `actor_rollout_ref.actor.entropy_checkpointing` | `false` | Checkpoint entropy computation |
| `actor_rollout_ref.actor.use_remove_padding` | From `model.use_remove_padding` | Remove padding |
| `actor_rollout_ref.actor.calculate_sum_pi_squared` | `false` | Compute sum of squared policy probabilities |
| `actor_rollout_ref.actor.sum_pi_squared_checkpointing` | `false` | Checkpoint squared-probability computation |
| `actor_rollout_ref.actor.use_fused_kernels` | From `model.use_fused_kernels` | Use fused kernels |

#### 1.1.3 Policy loss

| Parameter | Default | Description |
|--------|--------|------|
| `actor_rollout_ref.actor.policy_loss.loss_mode` | `vanilla` | Loss mode: vanilla, clip_cov, kl_cov, dppo_tv, dppo_kl, gspo, sapo, geo_mean, cispo, gpg, bypass_mode or reinforce_is |
| `actor_rollout_ref.actor.policy_loss.clip_cov_ratio` | `0.0002` | clip_cov covariance ratio |
| `actor_rollout_ref.actor.policy_loss.clip_cov_lb` | `1.0` | clip_cov lower bound |
| `actor_rollout_ref.actor.policy_loss.clip_cov_ub` | `5.0` | clip_cov upper bound |
| `actor_rollout_ref.actor.policy_loss.kl_cov_ratio` | `0.0002` | kl_cov covariance ratio |
| `actor_rollout_ref.actor.policy_loss.ppo_kl_coef` | `0.1` | PPO KL coefficient |

#### 1.1.4 Rollout

| Parameter | Default | Description |
|--------|--------|------|
| `actor_rollout_ref.rollout.name` | `???` | Required rollout engine name |
| `actor_rollout_ref.rollout.mode` | `async` | Rollout mode, such as async or sync |
| `actor_rollout_ref.rollout.nnodes` | `0` | Rollout node count |
| `actor_rollout_ref.rollout.n_gpus_per_node` | From `trainer.n_gpus_per_node` | GPUs per node |
| `actor_rollout_ref.rollout.temperature` | `1.0` | Sampling temperature |
| `actor_rollout_ref.rollout.top_k` | `-1` | Top-K; -1 disables it |
| `actor_rollout_ref.rollout.top_p` | `1` | Nucleus sampling probability |
| `actor_rollout_ref.rollout.prompt_length` | From `data.max_prompt_length` | Maximum prompt length |
| `actor_rollout_ref.rollout.response_length` | From `data.max_response_length` | Maximum response length |
| `actor_rollout_ref.rollout.dtype` | `bfloat16` | Inference dtype |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `0.5` | Inference device-memory fraction |
| `actor_rollout_ref.rollout.ignore_eos` | `false` | Ignore EOS |
| `actor_rollout_ref.rollout.enforce_eager` | `false` | Force PyTorch eager execution |
| `actor_rollout_ref.rollout.cudagraph_capture_sizes` | `null` | Graph capture sizes |
| `actor_rollout_ref.rollout.free_cache_engine` | `true` | Release cache after inference |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `2` | Inference TP degree |
| `actor_rollout_ref.rollout.data_parallel_size` | `1` | Inference DP degree |
| `actor_rollout_ref.rollout.expert_parallel_size` | `1` | Inference EP degree |
| `actor_rollout_ref.rollout.pipeline_model_parallel_size` | `1` | Inference PP degree |
| `actor_rollout_ref.rollout.max_num_batched_tokens` | `8192` | Maximum batched tokens per step |
| `actor_rollout_ref.rollout.max_model_len` | `null` | Maximum sequence length; null infers it |
| `actor_rollout_ref.rollout.max_num_seqs` | `1024` | Maximum concurrent sequences |
| `actor_rollout_ref.rollout.enable_chunked_prefill` | `true` | Chunked prefill |
| `actor_rollout_ref.rollout.enable_prefix_caching` | `true` | Prefix/KV caching |
| `actor_rollout_ref.rollout.logprobs_mode` | `processed_logprobs` | Log-probability mode |
| `actor_rollout_ref.rollout.scheduling_policy` | `fcfs` | Scheduling policy, such as fcfs |
| `actor_rollout_ref.rollout.load_format` | `dummy` | Weight loading format |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size` | `null` | Log-probability micro-batch size |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` | `null` | Log-probability micro-batch size per GPU |
| `actor_rollout_ref.rollout.log_prob_use_dynamic_bsz` | From `actor.use_dynamic_bsz` | Dynamic batching for log-probabilities |
| `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu` | From `actor.ppo_max_token_len_per_gpu` | Log-probability token limit per GPU |
| `actor_rollout_ref.rollout.disable_log_stats` | `true` | Disable inference statistics |
| `actor_rollout_ref.rollout.do_sample` | `true` | Sample; false selects greedy decoding |
| `actor_rollout_ref.rollout.n` | `1` | Responses per prompt |
| `actor_rollout_ref.rollout.over_sample_rate` | `0` | Oversampling rate |
| `actor_rollout_ref.rollout.multi_stage_wake_up` | `false` | Multi-stage wake-up |
| `actor_rollout_ref.rollout.calculate_log_probs` | `false` | Compute rollout log-probabilities |
| `actor_rollout_ref.rollout.skip_tokenizer_init` | `true` | Skip tokenizer initialization |
| `actor_rollout_ref.rollout.enable_rollout_routing_replay` | `false` | Record rollout routes for replay |
| `actor_rollout_ref.rollout.quantization` | `null` | Quantization method |
| `actor_rollout_ref.rollout.quantization_config_file` | `null` | Quantization configuration path |
| `actor_rollout_ref.rollout.layered_summon` | `false` | Layered summon, FSDP only |

#### 1.1.5 Validation sampling

| Parameter | Default | Description |
|--------|--------|------|
| `actor_rollout_ref.rollout.val_kwargs.top_k` | `-1` | Validation Top-K |
| `actor_rollout_ref.rollout.val_kwargs.top_p` | `1.0` | Validation Top-P |
| `actor_rollout_ref.rollout.val_kwargs.temperature` | `0` | Validation temperature; zero is greedy |
| `actor_rollout_ref.rollout.val_kwargs.n` | `1` | Validation responses per prompt |
| `actor_rollout_ref.rollout.val_kwargs.do_sample` | `false` | Sample during validation |

#### 1.1.6 Multi-turn interaction

| Parameter | Default | Description |
|--------|--------|------|
| `actor_rollout_ref.rollout.multi_turn.enable` | `false` | Enable multi-turn interaction |
| `actor_rollout_ref.rollout.multi_turn.max_assistant_turns` | `null` | Maximum assistant turns |
| `actor_rollout_ref.rollout.multi_turn.tool_config_path` | `null` | Tool configuration path |
| `actor_rollout_ref.rollout.multi_turn.max_user_turns` | `null` | Maximum user turns |
| `actor_rollout_ref.rollout.multi_turn.max_parallel_calls` | `1` | Maximum parallel tool calls |
| `actor_rollout_ref.rollout.multi_turn.max_tool_response_length` | `256` | Maximum tool response length |
| `actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side` | `middle` | Tool response truncation side |
| `actor_rollout_ref.rollout.multi_turn.interaction_config_path` | `null` | Interaction configuration path |
| `actor_rollout_ref.rollout.multi_turn.use_inference_chat_template` | `false` | Use inference chat template |
| `actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode` | `strict` | Tokenization consistency-check mode |
| `actor_rollout_ref.rollout.multi_turn.format` | `hermes` | Multi-turn format |
| `actor_rollout_ref.rollout.multi_turn.num_repeat_rollouts` | `null` | Repeated rollout count |

#### 1.1.7 Agents

| Parameter | Default | Description |
|--------|--------|------|
| `actor_rollout_ref.rollout.agent.num_workers` | `8` | Agent worker count |
| `actor_rollout_ref.rollout.agent.default_agent_loop` | `single_turn_agent` | Default agent loop |
| `actor_rollout_ref.rollout.agent.agent_loop_config_path` | `null` | Agent-loop configuration path |
| `actor_rollout_ref.rollout.agent.custom_async_server.path` | `null` | Custom async server path |
| `actor_rollout_ref.rollout.agent.custom_async_server.name` | `null` | Custom async server name |

#### 1.1.8 Checkpoint engine

| Parameter | Default | Description |
|--------|--------|------|
| `actor_rollout_ref.rollout.checkpoint_engine.backend` | `naive` | Checkpoint engine backend |
| `actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes` | `2048` | Weight-transfer bucket size in MiB |

#### 1.1.9 Tracing

| Parameter | Default | Description |
|--------|--------|------|
| `actor_rollout_ref.rollout.trace.project_name` | From `trainer.project_name` | Trace project name |
| `actor_rollout_ref.rollout.trace.experiment_name` | From `trainer.experiment_name` | Trace experiment name |
| `actor_rollout_ref.rollout.trace.backend` | `null` | Trace backend |
| `actor_rollout_ref.rollout.trace.token2text` | `false` | Decode tokens to text |
| `actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker` | `null` | Samples per step per worker |

#### 1.1.10 Prometheus

| Parameter | Default | Description |
|--------|--------|------|
| `actor_rollout_ref.rollout.prometheus.enable` | `false` | Enable Prometheus |
| `actor_rollout_ref.rollout.prometheus.port` | `9090` | Prometheus port |
| `actor_rollout_ref.rollout.prometheus.file` | `/tmp/ray/session_latest/metrics/prometheus/prometheus.yml` | Prometheus configuration path |
| `actor_rollout_ref.rollout.prometheus.served_model_name` | From `model.path` | Served model name |

#### 1.1.11 Reference model

| Parameter | Default | Description |
|--------|--------|------|
| `actor_rollout_ref.ref.rollout_n` | From `rollout.n` | Rollout count |
| `actor_rollout_ref.ref.strategy` | From `actor.strategy` | Training strategy |
| `actor_rollout_ref.ref.use_torch_compile` | From `actor.use_torch_compile` | Use torch.compile |
| `actor_rollout_ref.ref.log_prob_micro_batch_size` | `null` | Log-probability micro-batch size |
| `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` | `null` | Log-probability micro-batch size per GPU |
| `actor_rollout_ref.ref.log_prob_use_dynamic_bsz` | From `actor.use_dynamic_bsz` | Dynamic batching for log-probabilities |
| `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu` | From `actor.ppo_max_token_len_per_gpu` | Log-probability token limit per GPU |
| `actor_rollout_ref.ref.ulysses_sequence_parallel_size` | From `actor.ulysses_sequence_parallel_size` | Ulysses parallel degree |
| `actor_rollout_ref.ref.entropy_from_logits_with_chunking` | `false` | Chunk entropy computation from logits |
| `actor_rollout_ref.ref.entropy_checkpointing` | `false` | Checkpoint entropy computation |

#### 1.1.12 Critic optimizer

| Parameter | Default | Description |
|--------|--------|------|
| `critic.optim.lr` | `1.0e-05` | Critic learning rate |
| `critic.optim.lr_warmup_steps_ratio` | `0.0` | Warmup-step ratio |
| `critic.optim.total_training_steps` | `-1` | Total training steps |
| `critic.optim.weight_decay` | `0.01` | Weight decay |
| `critic.optim.lr_warmup_steps` | `-1` | Warmup steps |
| `critic.optim.betas` | `[0.9, 0.999]` | Adam moment coefficients |
| `critic.optim.clip_grad` | `1.0` | Gradient clipping threshold |
| `critic.optim.override_optimizer_config` | `null` / `{}` | Optimizer overrides |

#### 1.1.13 Critic policy

| Parameter | Default | Description |
|--------|--------|------|
| `critic.strategy` | `fsdp` / `megatron` | Training strategy |
| `critic.enable` | `null` | Enable critic; null selects automatically |
| `critic.ppo_mini_batch_size` | From `actor.ppo_mini_batch_size` | PPO mini-batch size |
| `critic.ppo_micro_batch_size` | `null` | PPO micro-batch size |
| `critic.ppo_micro_batch_size_per_gpu` | `null` | PPO micro-batch size per GPU |
| `critic.use_dynamic_bsz` | From `actor.use_dynamic_bsz` | Dynamic batching |
| `critic.ppo_max_token_len_per_gpu` | `32768` | PPO token limit per GPU |
| `critic.forward_max_token_len_per_gpu` | From `critic.ppo_max_token_len_per_gpu` | Forward token limit per GPU |
| `critic.ppo_epochs` | From `actor.ppo_epochs` | PPO update epochs |
| `critic.shuffle` | From `actor.shuffle` | Shuffle |
| `critic.data_loader_seed` | `42` / from `actor.data_loader_seed` | Data-loader seed |
| `critic.cliprange_value` | `0.5` | Value clipping range |
| `critic.loss_agg_mode` | From `actor.loss_agg_mode` | Loss aggregation mode |
| `critic.grad_clip` | `1.0` | Gradient clipping threshold |
| `critic.ulysses_sequence_parallel_size` | `1` | Ulysses parallel degree |
| `critic.forward_micro_batch_size` | From `critic.ppo_micro_batch_size` | Forward micro-batch size |
| `critic.forward_micro_batch_size_per_gpu` | From `critic.ppo_micro_batch_size_per_gpu` | Forward micro-batch size per GPU |

#### 1.1.14 Critic model

| Parameter | Default | Description |
|--------|--------|------|
| `critic.model.path` | `~/models/deepseek-llm-7b-chat` | Critic model path |
| `critic.model.tokenizer_path` | From `model.path` | Tokenizer path |
| `critic.model.override_config` | `{}` | Model configuration overrides |
| `critic.model.external_lib` | From `model.external_lib` | External library path |
| `critic.model.trust_remote_code` | From `model.trust_remote_code` | Trust remote model code |
| `critic.model.use_shm` | `false` | Use shared memory |
| `critic.model.enable_gradient_checkpointing` | `true` | Enable activation checkpointing |
| `critic.model.enable_activation_offload` | `false` | Enable activation offload |
| `critic.model.use_remove_padding` | `false` / `true` | Remove padding |
| `critic.model.lora_rank` | `0` | LoRA rank |
| `critic.model.lora_alpha` | `16` | LoRA alpha |
| `critic.model.target_modules` | `all-linear` | LoRA target modules |
| `critic.model.tiled_mlp.enabled` | `false` | Enable tiled MLP |
| `critic.model.tiled_mlp.num_shards` | `4` | MLP shard count |

#### 1.1.15 Data

| Parameter | Default | Description |
|--------|--------|------|
| `data.tokenizer` | `null` | Tokenizer path |
| `data.use_shm` | `false` | Use shared memory |
| `data.train_files` | `~/data/rlhf/gsm8k/train.parquet` | Training data paths |
| `data.val_files` | `~/data/rlhf/gsm8k/test.parquet` | Validation data paths |
| `data.train_max_samples` | `-1` | Maximum training samples; -1 is unlimited |
| `data.val_max_samples` | `-1` | Maximum validation samples |
| `data.prompt_key` | `prompt` | Prompt field |
| `data.reward_fn_key` | `data_source` | Reward-function routing field |
| `data.max_prompt_length` | `512` | Maximum prompt length |
| `data.max_response_length` | `512` | Maximum response length |
| `data.train_batch_size` | `1024` | Training batch size |
| `data.val_batch_size` | `null` | Validation batch size |
| `data.tool_config_path` | From `rollout.multi_turn.tool_config_path` | Tool configuration path |
| `data.return_raw_input_ids` | `false` | Return raw input IDs |
| `data.return_raw_chat` | `true` | Return raw chat |
| `data.return_full_prompt` | `false` | Return full prompt |
| `data.shuffle` | `true` | Shuffle training data |
| `data.seed` | `null` | Data-shuffle seed |
| `data.dataloader_num_workers` | `8` | Data-loader workers |
| `data.image_patch_size` | `14` | Image patch size |
| `data.validation_shuffle` | `false` | Shuffle validation data |
| `data.filter_overlong_prompts` | `false` | Filter overlong prompts |
| `data.filter_overlong_prompts_workers` | `1` | Overlong-prompt filter workers |
| `data.truncation` | `error` | Truncation policy |
| `data.image_key` | `images` | Image field |
| `data.video_key` | `videos` | Video field |
| `data.trust_remote_code` | `false` | Trust remote model code |
| `data.return_multi_modal_inputs` | `true` | Return multimodal inputs |

#### 1.1.16 Rewards

| Parameter | Default | Description |
|--------|--------|------|
| `reward.num_workers` | `8` | Reward workers |
| `reward.custom_reward_function.path` | `null` | Custom reward function path |
| `reward.custom_reward_function.name` | `compute_score` | Custom reward function name |
| `reward.reward_manager.source` | `register` | Reward-manager source |
| `reward.reward_manager.name` | `naive` | Reward-manager name |
| `reward.reward_model.enable` | `false` | Enable reward model |
| `reward.reward_model.enable_resource_pool` | `false` | Enable reward-model resource pool |
| `reward.reward_model.n_gpus_per_node` | `8` | Reward-model GPUs per node |
| `reward.reward_model.nnodes` | `0` | Reward-model nodes |
| `reward.reward_model.model_path` | `null` | Reward-model path |
| `reward.sandbox_fusion.url` | `null` | Sandbox Fusion URL |
| `reward.sandbox_fusion.max_concurrent` | `64` | Sandbox Fusion concurrency |
| `reward.sandbox_fusion.memory_limit_mb` | `1024` | Sandbox Fusion memory limit in MB |

#### 1.1.17 Algorithm

| Parameter | Default | Description |
|--------|--------|------|
| `algorithm.gamma` | `1.0` | Discount factor |
| `algorithm.lam` | `1.0` | GAE lambda |
| `algorithm.adv_estimator` | `gae` | Advantage estimator, such as gae |
| `algorithm.norm_adv_by_std_in_grpo` | `true` | Normalize GRPO advantages by standard deviation |
| `algorithm.use_kl_in_reward` | `false` | Add KL penalty to rewards |
| `algorithm.kl_penalty` | `kl` | KL penalty estimator |
| `algorithm.kl_ctrl.type` | `fixed` | KL controller, such as fixed or kl_adapter |
| `algorithm.kl_ctrl.kl_coef` | `0.001` | KL penalty coefficient |
| `algorithm.kl_ctrl.horizon` | `10000` | Adaptive KL horizon |
| `algorithm.kl_ctrl.target_kl` | `0.1` | Target KL |
| `algorithm.use_pf_ppo` | `false` | Enable PF-PPO |
| `algorithm.pf_ppo.reweight_method` | `pow` | PF-PPO reweighting method |
| `algorithm.pf_ppo.weight_pow` | `2.0` | PF-PPO weighting power |

#### 1.1.18 Rollout correction

| Parameter | Default | Description |
|--------|--------|------|
| `algorithm.rollout_correction.rollout_is` | `null` | Importance-sampling correction |
| `algorithm.rollout_correction.rollout_is_threshold` | `2.0` | IS weight threshold |
| `algorithm.rollout_correction.rollout_rs` | `null` | Rejection-sampling correction |
| `algorithm.rollout_correction.rollout_rs_threshold` | `null` | RS threshold |
| `algorithm.rollout_correction.bypass_mode` | `false` | Bypass mode |
| `algorithm.rollout_correction.loss_type` | `ppo_clip` | Correction loss type |
| `algorithm.rollout_correction.rollout_is_batch_normalize` | `false` | Batch-normalize IS weights |

#### 1.1.19 Trainer

| Parameter | Default | Description |
|--------|--------|------|
| `trainer.balance_batch` | `true` | Balance batches |
| `trainer.total_epochs` | `30` | Training epochs |
| `trainer.total_training_steps` | `null` | Total steps; null derives from epochs |
| `trainer.project_name` | `verl_examples` | Project name |
| `trainer.experiment_name` | `gsm8k` | Experiment name |
| `trainer.logger` | `[console, wandb]` | Logging backends |
| `trainer.log_val_generations` | `0` | Validation generations to log |
| `trainer.nnodes` | `1` | Training nodes |
| `trainer.n_gpus_per_node` | `8` | GPUs per node |
| `trainer.save_freq` | `-1` | Save frequency; -1 disables saves |
| `trainer.esi_redundant_time` | `0` | ESI time margin |
| `trainer.resume_mode` | `auto` | Resume mode, such as auto |
| `trainer.resume_from_path` | `null` | Resume path |
| `trainer.val_before_train` | `true` | Validate before training |
| `trainer.val_only` | `false` | Validation only |
| `trainer.test_freq` | `-1` | Validation frequency |
| `trainer.critic_warmup` | `0` | Critic warmup steps |
| `trainer.default_hdfs_dir` | `null` | Default HDFS directory |
| `trainer.del_local_ckpt_after_load` | `false` | Delete local checkpoint after loading |
| `trainer.default_local_dir` | `checkpoints/${trainer.project_name}/${trainer.experiment_name}` | Default local checkpoint directory |
| `trainer.max_actor_ckpt_to_keep` | `null` | Maximum retained actor checkpoints |
| `trainer.max_critic_ckpt_to_keep` | `null` | Maximum retained critic checkpoints |
| `trainer.ray_wait_register_center_timeout` | `300` | Ray registration timeout in seconds |
| `trainer.device` | `cuda` | Training device |
| `trainer.use_legacy_worker_impl` | `auto` | Use legacy workers |
| `trainer.rollout_data_dir` | `null` | Per-rollout output directory |

#### 1.1.20 Model

| Parameter | Default | Description |
|--------|--------|------|
| `actor_rollout_ref.model.path` | `~/models/deepseek-llm-7b-chat` | Model path |
| `actor_rollout_ref.model.hf_config_path` | `null` | Hugging Face configuration path |
| `actor_rollout_ref.model.tokenizer_path` | `null` | Tokenizer path |
| `actor_rollout_ref.model.use_shm` | `false` | Use shared memory |
| `actor_rollout_ref.model.trust_remote_code` | `false` | Trust remote model code |
| `actor_rollout_ref.model.custom_chat_template` | `null` | Custom chat template |
| `actor_rollout_ref.model.external_lib` | `null` | External library path |
| `actor_rollout_ref.model.override_config` | `{}` | Model configuration overrides |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | `true` | Enable activation checkpointing |
| `actor_rollout_ref.model.enable_activation_offload` | `false` | Enable activation offload |
| `actor_rollout_ref.model.use_remove_padding` | `true` / `false` | Remove padding |
| `actor_rollout_ref.model.lora_rank` | `0` | LoRA rank; zero disables LoRA |
| `actor_rollout_ref.model.lora_alpha` | `16` | LoRA alpha |
| `actor_rollout_ref.model.target_modules` | `all-linear` | LoRA target modules |
| `actor_rollout_ref.model.exclude_modules` | `null` | Excluded LoRA modules |
| `actor_rollout_ref.model.lora_adapter_path` | `null` | LoRA adapter path |
| `actor_rollout_ref.model.use_liger` | `false` | Use Liger kernels |
| `actor_rollout_ref.model.use_fused_kernels` | `false` | Use fused kernels |
| `actor_rollout_ref.model.fused_kernel_options.impl_backend` | `torch` | Fused-kernel backend |
| `actor_rollout_ref.model.tiled_mlp.enabled` | `false` | Enable tiled MLP |
| `actor_rollout_ref.model.tiled_mlp.num_shards` | `4` | MLP shard count |

#### 1.1.21 Shared engine

| Parameter | Default | Description |
|--------|--------|------|
| `actor_rollout_ref.hybrid_engine` | `true` | Hybrid engine sharing training/inference weights |
| `actor_rollout_ref.nccl_timeout` | `600` | NCCL timeout in seconds |
| `transfer_queue.enable` | `false` | Enable transfer queue |

---

### 1.2 FSDP parameters

These parameters belong to the FSDP _generated_ppo_trainer.yaml configuration.

#### 1.2.1 FSDP optimizer

| Parameter | Default | Description |
|--------|--------|------|
| `actor_rollout_ref.actor.optim.optimizer` | `AdamW` | Optimizer type |
| `actor_rollout_ref.actor.optim.optimizer_impl` | `torch.optim` | Optimizer implementation |
| `actor_rollout_ref.actor.optim.min_lr_ratio` | `0.0` | Minimum learning-rate ratio |
| `actor_rollout_ref.actor.optim.num_cycles` | `0.5` | Cosine schedule cycles |
| `actor_rollout_ref.actor.optim.lr_scheduler_type` | `constant` | Learning-rate scheduler |
| `actor_rollout_ref.actor.optim.zero_indexed_step` | `true` | Count steps from zero |
| `actor_rollout_ref.actor.optim.warmup_style` | `null` | Warmup style |

#### 1.2.2 Actor FSDP engine

| Parameter | Default | Description |
|--------|--------|------|
| `actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params` | `0` | Minimum parameter count for wrapping |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | `false` | CPU parameter offload |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | `false` | CPU optimizer offload |
| `actor_rollout_ref.actor.fsdp_config.offload_policy` | `false` | Offload policy |
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward` | `true` | Reshard after forward |
| `actor_rollout_ref.actor.fsdp_config.fsdp_size` | `-1` | FSDP group size; -1 selects global size |
| `actor_rollout_ref.actor.fsdp_config.forward_prefetch` | `false` | Forward parameter prefetch |
| `actor_rollout_ref.actor.fsdp_config.model_dtype` | `fp32` | Computation dtype |
| `actor_rollout_ref.actor.fsdp_config.use_orig_params` | `false` | Use original parameters |
| `actor_rollout_ref.actor.fsdp_config.seed` | `42` | Random seed |
| `actor_rollout_ref.actor.fsdp_config.full_determinism` | `false` | Full determinism |
| `actor_rollout_ref.actor.fsdp_config.forward_only` | `false` | Forward only; false for actor training |
| `actor_rollout_ref.actor.fsdp_config.strategy` | `fsdp` | Strategy type |
| `actor_rollout_ref.actor.fsdp_config.dtype` | `bfloat16` | Storage dtype |

#### 1.2.3 Reference FSDP engine

Same structure as the actor FSDP engine, with these differences:

| Parameter | Default | Description |
|--------|--------|------|
| `actor_rollout_ref.ref.fsdp_config.forward_only` | `true` | Reference forward-only execution |

Other defaults, including wrap_policy, offload, resharding, group size and dtype, match the actor FSDP engine.

#### 1.2.4 Critic FSDP engine

Same structure as the actor FSDP engine, with these differences:

| Parameter | Default | Description |
|--------|--------|------|
| `critic.model.fsdp_config.forward_only` | `false` | Train the critic |
| `critic.model.fsdp_config.use_remove_padding` | `false` | Keep critic padding |

Other defaults match the actor FSDP engine.

---

### 1.3 Megatron parameters

These parameters belong to _generated_ppo_megatron_trainer.yaml.

#### 1.3.1 Megatron optimizer

| Parameter | Default | Description |
|--------|--------|------|
| `actor_rollout_ref.actor.optim.optimizer` | `adam` | Optimizer type |
| `actor_rollout_ref.actor.optim.lr_warmup_init` | `0.0` | Initial warmup learning rate |
| `actor_rollout_ref.actor.optim.lr_decay_steps` | `null` | Learning-rate decay steps |
| `actor_rollout_ref.actor.optim.lr_decay_style` | `constant` | Decay style: constant, cosine, exponential, etc. |
| `actor_rollout_ref.actor.optim.min_lr` | `0.0` | Minimum learning rate |
| `actor_rollout_ref.actor.optim.weight_decay_incr_style` | `constant` | Weight-decay growth style |
| `actor_rollout_ref.actor.optim.lr_wsd_decay_style` | `exponential` | WSD learning-rate decay style |
| `actor_rollout_ref.actor.optim.lr_wsd_decay_steps` | `null` | WSD learning-rate decay steps |
| `actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler` | `false` | Use the checkpoint optimizer scheduler |

#### 1.3.2 Actor Megatron engine

| Parameter | Default | Description |
|--------|--------|------|
| `actor_rollout_ref.actor.megatron.param_offload` | `false` | CPU parameter offload |
| `actor_rollout_ref.actor.megatron.grad_offload` | `false` | CPU gradient offload |
| `actor_rollout_ref.actor.megatron.optimizer_offload` | `false` | CPU optimizer offload |
| `actor_rollout_ref.actor.megatron.tensor_model_parallel_size` | `1` | Tensor parallel degree |
| `actor_rollout_ref.actor.megatron.expert_model_parallel_size` | `1` | Expert parallel degree |
| `actor_rollout_ref.actor.megatron.expert_tensor_parallel_size` | `null` | Expert tensor parallel degree |
| `actor_rollout_ref.actor.megatron.pipeline_model_parallel_size` | `1` | Pipeline parallel degree |
| `actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size` | `null` | Virtual pipeline parallel degree |
| `actor_rollout_ref.actor.megatron.context_parallel_size` | `1` | Context parallel degree |
| `actor_rollout_ref.actor.megatron.sequence_parallel` | `true` | Sequence parallelism |
| `actor_rollout_ref.actor.megatron.use_distributed_optimizer` | `true` | Distributed optimizer |
| `actor_rollout_ref.actor.megatron.use_dist_checkpointing` | `false` | Distributed checkpointing |
| `actor_rollout_ref.actor.megatron.dist_checkpointing_path` | `null` | Distributed checkpoint path |
| `actor_rollout_ref.actor.megatron.dist_checkpointing_prefix` | `''` | Distributed checkpoint prefix |
| `actor_rollout_ref.actor.megatron.dist_ckpt_optim_fully_reshardable` | `false` | Fully reshardable optimizer checkpoint |
| `actor_rollout_ref.actor.megatron.distrib_optim_fully_reshardable_mem_efficient` | `false` | Memory-efficient optimizer resharding |
| `actor_rollout_ref.actor.megatron.seed` | `42` | Random seed |
| `actor_rollout_ref.actor.megatron.use_mbridge` | `true` | Bridge weight conversion |
| `actor_rollout_ref.actor.megatron.vanilla_mbridge` | `false` | Use deprecated mBridge; default uses Megatron-Bridge |
| `actor_rollout_ref.actor.megatron.use_remove_padding` | `true` | Remove padding |
| `actor_rollout_ref.actor.megatron.forward_only` | `false` | Forward-only execution |
| `actor_rollout_ref.actor.megatron.dtype` | `bfloat16` | Model dtype |
| `actor_rollout_ref.actor.megatron.load_weight` | `true` | Load weights |

#### 1.3.3 Transformer overrides

| Parameter | Default | Description |
|--------|--------|------|
| `override_transformer_config.recompute_granularity` | `null` | Recomputation granularity |
| `override_transformer_config.recompute_modules` | `[core_attn]` | Recomputed modules |
| `override_transformer_config.recompute_method` | `null` | Recomputation method |
| `override_transformer_config.recompute_num_layers` | `null` | Recomputed layer count |
| `override_transformer_config.attention_backend` | `flash` | Attention backend |

#### 1.3.4 Reference Megatron engine

Same structure as the actor Megatron engine, with these differences:

| Parameter | Default | Description |
|--------|--------|------|
| `actor_rollout_ref.ref.megatron.forward_only` | `true` | Reference forward-only execution |

Other defaults inherit actor Megatron settings, including param_offload and tensor_model_parallel_size.

#### 1.3.5 Critic Megatron engine

Same structure as the actor Megatron engine, with these differences:

| Parameter | Default | Description |
|--------|--------|------|
| `critic.megatron.forward_only` | `false` | Train the critic |

#### 1.3.6 Megatron LoRA

| Parameter | Default | Description |
|--------|--------|------|
| `model.lora.type` | `lora` | LoRA type |
| `model.lora.merge` | `false` | Merge LoRA weights |
| `model.lora.rank` | `0` | LoRA rank; zero disables it |
| `model.lora.alpha` | `32` | LoRA alpha |
| `model.lora.dropout` | `0.0` | LoRA dropout |
| `model.lora.target_modules` | `[linear_qkv, linear_proj, linear_fc1, linear_fc2]` | LoRA target modules |
| `model.lora.exclude_modules` | `[]` | Excluded LoRA modules |
| `model.lora.dropout_position` | `pre` | LoRA dropout position |
| `model.lora.lora_A_init_method` | `xavier` | LoRA A initialization |
| `model.lora.lora_B_init_method` | `zero` | LoRA B initialization |
| `model.lora.a2a_experimental` | `false` | Experimental all-to-all |
| `model.lora.dtype` | `null` | LoRA dtype |
| `model.lora.adapter_path` | `null` | LoRA adapter path |
| `model.lora.freeze_vision_model` | `true` | Freeze vision model |
| `model.lora.freeze_vision_projection` | `true` | Freeze vision projection |
| `model.lora.freeze_language_model` | `true` | Freeze language model |

#### 1.3.7 Megatron model overrides

| Parameter | Default | Description |
|--------|--------|------|
| `model.override_config.model_config` | `{}` | Model configuration overrides |
| `model.override_config.moe_config.freeze_moe_router` | `false` | Freeze MoE router |

#### 1.3.8 Megatron rollout layer mappings

| Parameter | Default | Description |
|--------|--------|------|
| `rollout.layer_name_map.qkv_layer_name` | `qkv` | QKV layer-name mapping |
| `rollout.layer_name_map.gate_proj_layer_name` | `gate_up` | Gate projection-name mapping |

---

### 1.4 Advanced parameters

#### 1.4.1 Profiler

| Parameter | Default | Description |
|--------|--------|------|
| `profiler.enable` | `false` | Enable profiler |
| `profiler.tool` | From `global_profiler.tool` | Profiler: nsys, npu, torch or torch_memory |
| `profiler.all_ranks` | `false` | Profile all ranks |
| `profiler.ranks` | `[]` | Ranks to profile |
| `profiler.save_path` | From `global_profiler.save_path` | Profiler output path |

#### 1.4.2 Global profiler

| Parameter | Default | Description |
|--------|--------|------|
| `global_profiler.tool` | `null` | Global profiler tool |
| `global_profiler.steps` | `null` | Steps to profile |
| `global_profiler.profile_continuous_steps` | `false` | Profile consecutive steps |
| `global_profiler.save_path` | `outputs/profile` | Global output path |

#### 1.4.3 Router replay

| Parameter | Default | Description |
|--------|--------|------|
| `router_replay.mode` | `disabled` | Route mode: disabled, record or replay |
| `router_replay.record_file` | `null` | Route recording path |
| `router_replay.replay_file` | `null` | Route replay path |

#### 1.4.4 Checkpoints

| Parameter | Default | Description |
|--------|--------|------|
| `checkpoint.save_contents` | `[model, optimizer, extra]` | Saved checkpoint contents |
| `checkpoint.load_contents` | From `checkpoint.save_contents` | Loaded checkpoint contents |
| `checkpoint.async_save` | `false` | Asynchronous checkpoint saving |
| `checkpoint.mbridge_config` | `{}` | mBridge configuration |

#### 1.4.5 QAT

| Parameter | Default | Description |
|--------|--------|------|
| `qat.enable` | `false` | Quantization-aware training |
| `qat.mode` | `w4a16` | Quantization mode |
| `qat.group_size` | `16` | Quantization group size |
| `qat.ignore_patterns` | `[lm_head, embed_tokens, re:.*mlp.gate$]` | Excluded quantization patterns |
| `qat.activation_observer` | `static_minmax` | Activation observer |
| `qat.quantization_config_path` | `null` | Quantization configuration path |

#### 1.4.6 MTP

| Parameter | Default | Description |
|--------|--------|------|
| `mtp.enable` | `false` | Multi-token prediction |
| `mtp.enable_train` | `false` | Enable MTP training |
| `mtp.enable_rollout` | `false` | Enable MTP rollout |
| `mtp.detach_encoder` | `false` | Detach encoder gradients |
| `mtp.mtp_loss_scaling_factor` | `0.1` | MTP loss scaling factor |
| `mtp.speculative_algorithm` | `EAGLE` | Speculative decoding algorithm |
| `mtp.speculative_num_steps` | `3` | Speculative steps |
| `mtp.speculative_eagle_topk` | `1` | EAGLE Top-K |
| `mtp.speculative_num_draft_tokens` | `4` | Draft-token count |
| `mtp.method` | `mtp` | MTP method |
| `mtp.num_speculative_tokens` | `1` | Speculative-token count |

---

## 2. Training metrics

Metrics emitted during RL iterations:

### 2.1 Progress

| Metric | Description |
|------|------|
| `training/global_step` | Global training step |
| `training/epoch` | Training epoch |

### 2.2 Actor

| Metric | Description |
|------|------|
| `actor/pg_loss` | Policy-gradient objective, such as PPO clipped loss |
| `actor/kl_loss` | Current/reference KL loss, when use_kl_loss=True |
| `actor/entropy` | Policy entropy, when calculate_entropy=True or entropy_coeff is nonzero |
| `actor/grad_norm` | Reported actor gradient norm; clipping semantics depend on the backend |
| `actor/lr` | Actor learning rate |
| `actor/pg_clipfrac` | Fraction affected by PPO clipping |
| `actor/ppo_kl` | Estimated current/old-policy KL |
| `actor/pg_clipfrac_lower` | Lower-clipping fraction for applicable loss modes |
| `actor/reward_kl_penalty` | Mean reward KL penalty, when use_kl_in_reward=True |
| `actor/reward_kl_penalty_coeff` | Reward KL coefficient beta |
| `actor/kl_coef` | KL loss coefficient, when use_kl_loss=True |

### 2.3 Critic

| Metric | Description |
|------|------|
| `critic/vf_loss` | Value-function loss |
| `critic/vf_clipfrac` | Value-clipping fraction |
| `critic/vpred_mean` | Mean predicted value |
| `critic/grad_norm` | Reported critic gradient norm |
| `critic/lr` | Critic learning rate |
| `critic/vf_explained_var` | Explained variance: 1 - Var(returns-values)/Var(returns), with an enabled critic |

### 2.4 Data statistics

| Metric | Description |
|------|------|
| `critic/score/mean` | Mean non-aborted sequence score |
| `critic/score/max` | Maximum non-aborted sequence score |
| `critic/score/min` | Minimum non-aborted sequence score |
| `critic/rewards/mean` | Mean non-aborted sequence reward |
| `critic/rewards/max` | Maximum non-aborted sequence reward |
| `critic/rewards/min` | Minimum non-aborted sequence reward |
| `critic/advantages/mean` | Mean valid-token advantage |
| `critic/advantages/max` | Maximum valid-token advantage |
| `critic/advantages/min` | Minimum valid-token advantage |
| `critic/returns/mean` | Mean valid-token return |
| `critic/returns/max` | Maximum valid-token return |
| `critic/returns/min` | Minimum valid-token return |
| `critic/values/mean` | Mean valid-token critic value, with critic enabled |
| `critic/values/max` | Maximum valid-token critic value, with critic enabled |
| `critic/values/min` | Minimum valid-token critic value, with critic enabled |
| `response_length/mean` | Mean response length, including aborted samples |
| `response_length/max` | Maximum response length |
| `response_length/min` | Minimum response length |
| `response_length/clip_ratio` | Fraction of responses reaching the length limit |
| `response_length_non_aborted/mean` | Mean non-aborted response length |
| `response_length_non_aborted/max` | Maximum non-aborted response length |
| `response_length_non_aborted/min` | Minimum non-aborted response length |
| `response_length_non_aborted/clip_ratio` | Fraction of non-aborted responses reaching the limit |
| `response/aborted_ratio` | Aborted fraction, defined here by zero response length |
| `prompt_length/mean` | Mean prompt length |
| `prompt_length/max` | Maximum prompt length |
| `prompt_length/min` | Minimum prompt length |
| `prompt_length/clip_ratio` | Fraction of prompts reaching the length limit |
| `num_turns/mean` | Mean interaction turns in multi-turn runs |
| `num_turns/max` | Maximum interaction turns |
| `num_turns/min` | Minimum interaction turns |
| `tool_call_counts/mean` | Mean tool calls, when tool_call_counts exists |
| `tool_call_counts/max` | Maximum tool calls |
| `tool_call_counts/min` | Minimum tool calls |

### 2.5 Timing

| Metric | Description |
|------|------|
| `timing_s/gen` | Generation time in seconds |
| `timing_s/ref` | Reference log-probability time in seconds |
| `timing_s/values` | Critic value computation time in seconds |
| `timing_s/adv` | Advantage computation time in seconds |
| `timing_s/update_critic` | Critic update time in seconds |
| `timing_s/update_actor` | Actor update time in seconds |
| `timing_s/step` | Total step time in seconds |
| `timing_s/old_log_prob` | Old-policy log-probability time in seconds |
| `timing_s/reward` | Reward computation time in seconds |
| `timing_s/testing` | Validation time in seconds |
| `timing_s/save_checkpoint` | Checkpoint saving time in seconds |
| `timing_s/update_weights` | Weight synchronization time in seconds |
| `timing_per_token_ms/gen` | Generation milliseconds per token |
| `timing_per_token_ms/ref` | Reference milliseconds per token |
| `timing_per_token_ms/values` | Critic value milliseconds per token |
| `timing_per_token_ms/adv` | Advantage milliseconds per token |
| `timing_per_token_ms/update_critic` | Critic update milliseconds per token |
| `timing_per_token_ms/update_actor` | Actor update milliseconds per token |

### 2.6 Performance

| Metric | Description |
|------|------|
| `perf/total_num_tokens` | Total tokens processed this step |
| `perf/time_per_step` | Step duration in seconds |
| `perf/throughput` | Tokens / (time × device count) |
| `perf/max_memory_allocated_gb` | Maximum allocated device memory in GB |
| `perf/max_memory_reserved_gb` | Maximum reserved device memory in GB |
| `perf/cpu_memory_used_gb` | Used CPU memory in GB |
| `perf/mfu/actor` | Actor training model FLOPs utilization |
| `perf/mfu/critic` | Critic training model FLOPs utilization |
| `perf/mfu/actor_infer` | Actor inference model FLOPs utilization |

### 2.7 Variance proxies

| Metric | Description |
|------|------|
| `variance_proxy/proxy1_signal_strength` | Signal: squared norm of mean gradient |
| `variance_proxy/proxy2_total_power` | Total power: expected squared gradient norm |
| `variance_proxy/proxy3_pure_noise` | Noise proxy: (Proxy2 - Proxy1)/(N-1) |
| `variance_proxy/expected_a_squared` | Expected squared advantage E[A^2] |
| `variance_proxy/expected_w` | Expected W-score proxy E[W] |

### 2.8 Conditional metrics

The following appear only under their stated conditions:

#### 2.8.1 Rollout correction

Enabled by rollout_correction, with rollout_corr/ prefixes.

**IS weights**, when importance correction is enabled:

| Metric | Description |
|------|------|
| `rollout_corr/rollout_is_mean` | Mean IS weight |
| `rollout_corr/rollout_is_max` | Maximum IS weight |
| `rollout_corr/rollout_is_min` | Minimum IS weight |
| `rollout_corr/rollout_is_std` | IS weight standard deviation |
| `rollout_corr/rollout_is_ratio_fraction_high` | Fraction above the upper IS threshold |
| `rollout_corr/rollout_is_ratio_fraction_low` | Fraction below the lower IS threshold |
| `rollout_corr/rollout_is_eff_sample_size` | Effective sample size |
| `rollout_corr/rollout_is_seq_mean` | Mean sequence-level IS weight |
| `rollout_corr/rollout_is_seq_std` | Sequence-level IS standard deviation |
| `rollout_corr/rollout_is_seq_max` | Maximum sequence-level IS weight |
| `rollout_corr/rollout_is_seq_min` | Minimum sequence-level IS weight |
| `rollout_corr/rollout_is_seq_max_deviation` | Maximum sequence-level deviation from 1.0 |
| `rollout_corr/rollout_is_seq_fraction_high` | Sequence-level fraction above the upper threshold |
| `rollout_corr/rollout_is_seq_fraction_low` | Sequence-level fraction below the lower threshold |
| `rollout_corr/rollout_is_batch_norm_factor` | Batch normalization factor, when rollout_is_batch_normalize=True |

**Rejection sampling**, when RS correction is enabled:

| Metric | Description |
|------|------|
| `rollout_corr/rollout_rs_{option}_mean` | Mean RS statistic |
| `rollout_corr/rollout_rs_{option}_max` | Maximum RS statistic |
| `rollout_corr/rollout_rs_{option}_min` | Minimum RS statistic |
| `rollout_corr/rollout_rs_{option}_std` | RS standard deviation |
| `rollout_corr/rollout_rs_{option}_fraction_high` | Fraction above the upper threshold |
| `rollout_corr/rollout_rs_{option}_fraction_low` | Fraction below the lower threshold |
| `rollout_corr/rollout_rs_{option}_seq_mean` | Mean sequence-level RS statistic |
| `rollout_corr/rollout_rs_{option}_seq_std` | Sequence-level RS standard deviation |
| `rollout_corr/rollout_rs_{option}_seq_max` | Maximum sequence-level RS statistic |
| `rollout_corr/rollout_rs_{option}_seq_min` | Minimum sequence-level RS statistic |
| `rollout_corr/rollout_rs_{option}_seq_max_deviation` | Maximum sequence-level RS deviation from zero |
| `rollout_corr/rollout_rs_{option}_seq_fraction_high` | Sequence-level fraction above the upper threshold |
| `rollout_corr/rollout_rs_{option}_seq_fraction_low` | Sequence-level fraction below the lower threshold |
| `rollout_corr/rollout_rs_{option}_masked_fraction` | Masked-token fraction |
| `rollout_corr/rollout_rs_{option}_seq_masked_fraction` | Masked-sequence fraction |
| `rollout_corr/rollout_rs_masked_fraction` | Overall masked-token fraction |
| `rollout_corr/rollout_rs_seq_masked_fraction` | Overall masked-sequence fraction |

**Off-policy diagnostics**, when enabled:

| Metric | Description |
|------|------|
| `rollout_corr/training_ppl` | Training-policy perplexity |
| `rollout_corr/training_log_ppl` | Training-policy log perplexity |
| `rollout_corr/kl` | Direct rollout-to-training KL estimate |
| `rollout_corr/k3_kl` | K3 KL estimate |
| `rollout_corr/rollout_ppl` | Rollout-policy perplexity |
| `rollout_corr/rollout_log_ppl` | Rollout-policy log perplexity |
| `rollout_corr/log_ppl_diff` | Log-perplexity difference: rollout minus training |
| `rollout_corr/log_ppl_abs_diff` | Mean absolute log-perplexity difference |
| `rollout_corr/log_ppl_diff_max` | Maximum log-perplexity difference |
| `rollout_corr/log_ppl_diff_min` | Minimum log-perplexity difference |
| `rollout_corr/ppl_ratio` | training_ppl / rollout_ppl |
| `rollout_corr/chi2_token` | Token-level chi-squared divergence |
| `rollout_corr/chi2_seq` | Sequence-level chi-squared divergence |

#### 2.8.2 Sequence balancing

Enabled by balance_batch:

| Metric | Description |
|------|------|
| `global_seqlen/min` | Minimum DP-partition total length before balancing |
| `global_seqlen/max` | Maximum DP-partition total length before balancing |
| `global_seqlen/minmax_diff` | Pre-balancing maximum minus minimum |
| `global_seqlen/balanced_min` | Minimum partition total after balancing |
| `global_seqlen/balanced_max` | Maximum partition total after balancing |
| `global_seqlen/mean` | Mean partition total length |

#### 2.8.3 GDPO rewards

Emitted with the GDPO estimator:

| Metric | Description |
|------|------|
| `gdpo/{key}/mean` | Mean of each GDPO reward component |
| `gdpo/{key}/std` | Standard deviation of each component |
| `gdpo/{key}/max` | Maximum of each component |
| `gdpo/{key}/min` | Minimum of each component |

#### 2.8.4 Training/rollout agreement

Enabled by actor_rollout_ref.rollout.calculate_log_probs=True:

| Metric | Description |
|------|------|
| `training/rollout_probs_diff_valid` | Validity marker, equal to 1 |
| `training/rollout_probs_diff_max` | Maximum rollout/actor probability difference |
| `training/rollout_probs_diff_mean` | Mean rollout/actor probability difference |
| `training/rollout_probs_diff_std` | Standard deviation of probability differences |
| `training/rollout_actor_probs_pearson_corr` | Pearson correlation of rollout/actor probabilities |

#### 2.8.5 Validation

Emitted during validation:

| Metric | Description |
|------|------|
| `val-core/{data_source}/{var_name}/{metric_name}` | Core metrics, such as mean@N, maj@N and best@N |
| `val-aux/{data_source}/{var_name}/{metric_name}` | Auxiliary metrics, such as std@N and worst@N |
| `val-aux/num_turns/mean` | Mean validation interaction turns |
| `val-aux/num_turns/max` | Maximum validation interaction turns |
| `val-aux/num_turns/min` | Minimum validation interaction turns |
