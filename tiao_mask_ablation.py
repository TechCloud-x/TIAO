#!/usr/bin/env python3
"""Run TIAO source-mask probability ablations on CNN/DailyMail."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
from transformers.integrations.deepspeed import (
    set_hf_deepspeed_config,
    unset_hf_deepspeed_config,
)

from tiao import (
    ModelArguments,
    REPOSITORY_ROOT,
    _completion_text,
    _is_main_process,
    _load_cnn_dailymail_split,
    _local_or_remote,
)
from tiao_mask_ablation_trainer import (
    TIAOMaskAblationConfig,
    TIAOMaskAblationTrainer,
)
from unieval import SumEvaluator
from utils import convert_to_json


def _probability_slug(probability: float) -> str:
    """Return a stable filesystem label for an arbitrary valid probability."""

    return format(probability, ".12g").replace(".", "p")


def main() -> None:
    parser = HfArgumentParser((TIAOMaskAblationConfig, ModelArguments))
    training_args, model_args = parser.parse_args_into_dataclasses()
    is_main = _is_main_process()
    set_seed(model_args.random_seed)

    model_name = Path(model_args.base_model_name_or_path.rstrip("/\\")).name
    probability_slug = _probability_slug(training_args.source_mask_probability)
    final_model_directory = Path(
        model_args.final_model_output_dir
        or REPOSITORY_ROOT
        / "outputs"
        / "tiao-source-mask-ablation"
        / f"mask-{probability_slug}"
        / model_name
        / "final-model"
    )
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    if is_main:
        print("=" * 72)
        print("TIAO source-mask ablation training configuration")
        print("=" * 72)
        print(training_args)
        print(json.dumps(model_args.__dict__, indent=2))
        print(f"Final model directory: {final_model_directory}")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    evaluator_device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    evaluator_deepspeed = model_args.unieval_model_deepspeed_config
    if evaluator_deepspeed and evaluator_deepspeed.lower() in {
        "none",
        "null",
        "false",
    }:
        evaluator_deepspeed = None

    deepspeed_context = getattr(training_args, "hf_deepspeed_config", None)
    if deepspeed_context is not None:
        unset_hf_deepspeed_config()
    try:
        evaluator = SumEvaluator(
            max_length=model_args.unieval_max_length,
            device=evaluator_device,
            deepspeed_config=evaluator_deepspeed,
            unieval_model_name_or_path=model_args.unieval_model_name_or_path,
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            show_progress=is_main,
        )
    finally:
        if deepspeed_context is not None:
            set_hf_deepspeed_config(deepspeed_context)

    example_printed = False

    def evaluate_dimension(
        completions,
        origin_text,
        ref_text,
        dimension: str,
    ) -> list[float]:
        nonlocal example_printed
        summaries = [_completion_text(completion) for completion in completions]
        records = convert_to_json(summaries, origin_text, ref_text)
        scores = evaluator.evaluate(records, dims=[dimension])
        if is_main and not example_printed and summaries:
            print(f"Generated summary example:\n{summaries[0]}")
            print(f"Reference summary example:\n{ref_text[0]}")
            print(f"UniEval example: {scores[0]}")
            example_printed = True
        return [float(item[dimension]) for item in scores]

    def coherence_reward(completions, origin_text, ref_text, **kwargs):
        return evaluate_dimension(completions, origin_text, ref_text, "coherence")

    def consistency_reward(completions, origin_text, ref_text, **kwargs):
        return evaluate_dimension(completions, origin_text, ref_text, "consistency")

    def fluency_reward(completions, origin_text, ref_text, **kwargs):
        return evaluate_dimension(completions, origin_text, ref_text, "fluency")

    def relevance_reward(completions, origin_text, ref_text, **kwargs):
        return evaluate_dimension(completions, origin_text, ref_text, "relevance")

    def repetition_reward(completions, origin_text, ref_text, **kwargs):
        summaries = [_completion_text(completion) for completion in completions]
        rewards: list[float] = []
        for summary in summaries:
            words = summary.split()
            if len(words) < 2:
                rewards.append(0.0)
                continue
            bigrams = list(zip(words, words[1:]))
            rewards.append(len(set(bigrams)) / len(bigrams))
        return rewards

    train_dataset = _load_cnn_dailymail_split(
        model_args.dataset_path,
        "train",
        model_args.dataset_sample_train_num,
        model_args.random_seed,
    )
    validation_dataset = _load_cnn_dailymail_split(
        model_args.dataset_path,
        "validation",
        model_args.dataset_sample_eval_num,
        model_args.random_seed,
    )

    local_model = _local_or_remote(model_args.base_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=local_model,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.base_model_name_or_path,
        local_files_only=local_model,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main:
        print(f"Distributed processes: {os.environ.get('WORLD_SIZE', '1')}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(validation_dataset)}")

    trainer = TIAOMaskAblationTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            coherence_reward,
            consistency_reward,
            fluency_reward,
            relevance_reward,
            repetition_reward,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    trainer.accelerator.wait_for_everyone()
    trainer.save_model(str(final_model_directory))
    if trainer.is_world_process_zero():
        final_model_directory.mkdir(parents=True, exist_ok=True)
        trainer.state.save_to_json(str(final_model_directory / "trainer_state.json"))
        print(f"Final model saved to {final_model_directory}", flush=True)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
