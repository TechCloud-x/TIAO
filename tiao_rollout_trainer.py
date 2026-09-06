"""GRPO rollout implementation used by the standalone TIAO trainer."""

from contextlib import contextmanager

from trl.trainer.grpo_trainer import *  # noqa: F403


@contextmanager
def _generation_eval_mode(model):
    """Temporarily disable training-only checkpointing while sampling."""

    was_training = model.training
    try:
        model.eval()
        yield model
    finally:
        model.train(was_training)


class TIAORolloutTrainer(GRPOTrainer):  # noqa: F405
    """Preserve the validated distributed generation and reward pipeline."""

    _tag_names = ["trl", "tiao"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],  # noqa: F405
        reward_funcs: Union[RewardFunc, list[RewardFunc]],  # noqa: F405
        args: Optional[GRPOConfig] = None,  # noqa: F405
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,  # noqa: F405
        eval_dataset: Optional[  # noqa: F405
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,  # noqa: F405
        reward_processing_classes: Optional[  # noqa: F405
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,  # noqa: F405
        optimizers: tuple[
            Optional[torch.optim.Optimizer],  # noqa: F405
            Optional[torch.optim.lr_scheduler.LambdaLR],  # noqa: F405
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,  # noqa: F405
        only_unieval: str = "false",
    ) -> None:
        self.only_unieval = only_unieval
        super().__init__(
            model,
            reward_funcs,
            args,
            train_dataset,
            eval_dataset,
            processing_class,
            reward_processing_classes,
            callbacks,
            optimizers,
            peft_config,
        )

    def _generate_and_score_completions(
        self,
        inputs: list[dict[str, Union[torch.Tensor, Any]]],  # noqa: F405
    ) -> dict[str, Union[torch.Tensor, Any]]:  # noqa: F405
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [item["prompt"] for item in inputs]
        prompts_text = [
            maybe_apply_chat_template(item, self.processing_class)["prompt"]  # noqa: F405
            for item in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            prompts_text = self.processing_class.batch_decode(
                prompt_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            prompts_text = [
                re.sub(  # noqa: F405
                    rf"^({re.escape(self.processing_class.pad_token)})+",  # noqa: F405
                    "",
                    text,
                )
                for text in prompts_text
            ]

        if self.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)  # noqa: F405
                if self.accelerator.is_main_process:
                    unique_prompts = all_prompts_text[:: self.num_generations]
                    with profiling_context(self, "vLLM.generate"):  # noqa: F405
                        completion_ids = self.vllm_client.generate(
                            prompts=unique_prompts,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                completion_ids = broadcast_object_list(  # noqa: F405
                    completion_ids,
                    from_process=0,
                )
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(  # noqa: F405
                        backend="outlines",
                        regex=self.guided_decoding_regex,
                    )
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "guided_decoding": guided_decoding,
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)  # noqa: F405

                if self.vllm_tensor_parallel_size > 1:
                    local_prompt_count = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(  # noqa: F405
                        gathered_prompts,
                        prompts_text,
                        group=self.tp_group,
                    )
                    all_prompts_text = [
                        prompt for group_prompts in gathered_prompts for prompt in group_prompts
                    ]
                else:
                    all_prompts_text = prompts_text

                with profiling_context(self, "vLLM.generate"):  # noqa: F405
                    all_outputs = self.llm.generate(
                        all_prompts_text,
                        sampling_params=sampling_params,
                        use_tqdm=False,
                    )
                completion_ids = [
                    output.token_ids
                    for outputs in all_outputs
                    for output in outputs.outputs
                ]

                if self.vllm_tensor_parallel_size > 1:
                    # Every rank generates the tensor-parallel group's outputs;
                    # retain only the slice assigned to this rank.
                    local_rank = torch.distributed.get_rank(group=self.tp_group)  # noqa: F405
                    tp_slice = slice(
                        local_rank * local_prompt_count,
                        (local_rank + 1) * local_prompt_count,
                    )
                    completion_ids = completion_ids[tp_slice]

            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]  # noqa: F405
            completion_ids = pad(  # noqa: F405
                completion_ids,
                padding_value=self.processing_class.pad_token_id,
            )
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # noqa: F405
        else:
            # Evaluation retains ZeRO-3 hooks so parameters are materialized
            # layer by layer. Training uses the validated full-gather path.
            with unwrap_model_for_generation(  # noqa: F405
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=(
                    self.args.ds3_gather_for_generation and mode == "train"
                ),
            ) as unwrapped_model:
                with (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)  # noqa: F405
                    if self.is_fsdp_enabled
                    else nullcontext()  # noqa: F405
                ):
                    # Autoregressive sampling needs no gradients. Evaluation
                    # mode also avoids gradient-checkpointing and KV-cache conflicts.
                    with _generation_eval_mode(unwrapped_model), torch.no_grad():  # noqa: F405
                        prompt_completion_ids = unwrapped_model.generate(
                            prompt_ids,
                            attention_mask=prompt_mask,
                            generation_config=self.generation_config,
                        )

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_index = torch.full(  # noqa: F405
            (is_eos.size(0),),
            is_eos.size(1),
            dtype=torch.long,  # noqa: F405
            device=device,
        )
        eos_index[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(  # noqa: F405
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_index.unsqueeze(1)).int()
        completion_ids_list = [
            [token_id.item() for token_id, keep in zip(row, mask_row) if keep]
            for row, mask_row in zip(completion_ids, completion_mask)
        ]
        completion_lengths = completion_mask.sum(1)

        if self.mask_truncated_completions:
            truncated = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated).unsqueeze(1).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # noqa: F405
        logits_to_keep = completion_ids.size(1)
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        with torch.no_grad():  # noqa: F405
            if (
                self.num_iterations > 1
                or self.args.steps_per_generation > self.args.gradient_accumulation_steps
            ):
                old_per_token_logps = self._get_per_token_logps(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )
            else:
                old_per_token_logps = None

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                        )
            else:
                ref_per_token_logps = None

        completions_text = self.processing_class.batch_decode(
            completion_ids,
            skip_special_tokens=True,
        )
        if is_conversational(inputs[0]):  # noqa: F405
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = self._calculate_rewards(
            inputs,
            prompts,
            completions,
            completion_ids_list,
        )
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        # Keep an additive reward view for interpretable training diagnostics.
        if self.only_unieval.lower() == "true":
            diagnostic_rewards = (
                rewards_per_func[:, :4]
                * self.reward_weights[:4].to(device).unsqueeze(0)
            ).nansum(dim=1)
        else:
            diagnostic_rewards = rewards
        diagnostic_means = diagnostic_rewards.view(-1, self.num_generations).mean(dim=1)
        diagnostic_stds = diagnostic_rewards.view(-1, self.num_generations).std(dim=1)
        diagnostic_std_is_zero = torch.isclose(  # noqa: F405
            diagnostic_stds,
            torch.zeros_like(diagnostic_stds),  # noqa: F405
        )
        diagnostic_means = diagnostic_means.repeat_interleave(self.num_generations, dim=0)
        diagnostic_stds = diagnostic_stds.repeat_interleave(self.num_generations, dim=0)

        grouped_means = rewards.view(-1, self.num_generations).mean(dim=1)
        grouped_stds = rewards.view(-1, self.num_generations).std(dim=1)
        grouped_means = grouped_means.repeat_interleave(self.num_generations, dim=0)
        grouped_stds = grouped_stds.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - grouped_means
        if self.scale_rewards:
            advantages = advantages / (grouped_stds + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        if mode == "train":
            self.state.num_input_tokens_seen += (
                self.accelerator.gather(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        aggregate_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(
            aggregate_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_length"].append(
            aggregate_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_length"].append(
            aggregate_lengths.float().max().item()
        )

        terminated = self.accelerator.gather(is_eos.any(dim=1))
        terminated_lengths = aggregate_lengths[terminated]
        clipped_ratio = 1 - len(terminated_lengths) / len(aggregate_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_ratio)
        if len(terminated_lengths) == 0:
            terminated_lengths = torch.zeros(1, device=device)  # noqa: F405
        self._metrics[mode]["completions/mean_terminated_length"].append(
            terminated_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_terminated_length"].append(
            terminated_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_terminated_length"].append(
            terminated_lengths.float().max().item()
        )

        for index, reward_name in enumerate(self.reward_func_names):
            self._metrics[mode][f"rewards/{reward_name}/mean"].append(
                torch.nanmean(rewards_per_func[:, index]).item()  # noqa: F405
            )
            self._metrics[mode][f"rewards/{reward_name}/std"].append(
                nanstd(rewards_per_func[:, index]).item()  # noqa: F405
            )
        self._metrics[mode]["reward"].append(diagnostic_means.mean().item())
        self._metrics[mode]["reward_std"].append(diagnostic_stds.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(
            diagnostic_std_is_zero.float().mean().item()
        )

        self._textual_logs["prompt"].extend(gather_object(prompts_text))  # noqa: F405
        self._textual_logs["completion"].extend(gather_object(completions_text))  # noqa: F405
        for index, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, index].tolist())
        self._textual_logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }

