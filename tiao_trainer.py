"""Token Importance-Aware Optimization for text summarization.

TIAO masks half of the source tokens independently for each sampled completion
and teacher-forces the same completion under the complete and masked prompts.
For every sampled completion token it computes the low-variance estimator

    d_t = log policy_masked(y_t) - log policy_complete(y_t)
    importance_t = exp(d_t) - d_t - 1

The detached importance is used for credit assignment, not as a loss penalty.
Trajectory importance is the mean of valid token scores. It rescales the GRPO
advantage by its ratio to the global rollout mean, preserving a mean scale of
one. The clipped policy surrogate is retained only for the top
``ceil(0.4 * valid_token_count)`` tokens in each trajectory. No
reference-policy KL penalty is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.distributed as dist
from trl import GRPOConfig
from trl.data_utils import maybe_apply_chat_template

from tiao_rollout_trainer import TIAORolloutTrainer


_TEXT_MASK_PROBABILITY = 0.5
_TOKEN_GATE_KEEP_RATIO = 0.4
_LOG_RATIO_CLAMP = 20.0
_IMPORTANCE_CLAMP_MAX = 10.0
_QWEN_FIM_MASK_TOKEN = "<|fim_pad|>"


@dataclass
class TIAOConfig(GRPOConfig):
    """TRL 0.19.1-compatible configuration for TIAO."""

    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a DeepSpeed JSON configuration file."},
    )
    beta: float = field(
        default=0.0,
        metadata={"help": "Reference-policy KL coefficient; TIAO requires 0."},
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.beta != 0.0:
            raise ValueError("TIAO requires --beta 0")
        if self.loss_type not in {"grpo", "bnpo", "dr_grpo"}:
            raise ValueError(
                "TIAO supports loss_type=grpo, bnpo, or dr_grpo"
            )


def _low_variance_token_kl(
    complete_logps: torch.Tensor,
    masked_logps: torch.Tensor,
) -> torch.Tensor:
    """Estimate KL(complete context || masked context) per sampled token."""

    log_ratio = (masked_logps.float() - complete_logps.float()).clamp(
        min=-_LOG_RATIO_CLAMP,
        max=_LOG_RATIO_CLAMP,
    )
    importance = torch.expm1(log_ratio) - log_ratio
    return torch.nan_to_num(
        importance,
        nan=0.0,
        posinf=_IMPORTANCE_CLAMP_MAX,
        neginf=0.0,
    ).clamp_(min=0.0, max=_IMPORTANCE_CLAMP_MAX)


def _exact_top_fraction_mask(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    keep_ratio: float = _TOKEN_GATE_KEEP_RATIO,
) -> torch.Tensor:
    """Select exactly ceil(keep_ratio * valid tokens) positions in each row."""

    valid = valid_mask.bool()
    valid_counts = valid.sum(dim=1)
    keep_counts = torch.ceil(valid_counts.float() * keep_ratio).to(torch.long)
    keep_counts = torch.where(
        valid_counts > 0,
        keep_counts.clamp(min=1),
        keep_counts,
    )
    sortable_scores = scores.float().masked_fill(~valid, -torch.inf)
    sorted_indices = torch.argsort(sortable_scores, dim=1, descending=True)
    ranks = torch.arange(scores.size(1), device=scores.device).expand_as(sorted_indices)
    sorted_keep = ranks < keep_counts.unsqueeze(1)
    gate = torch.zeros_like(valid)
    gate.scatter_(1, sorted_indices, sorted_keep)
    return gate & valid


class TIAOTrainer(TIAORolloutTrainer):
    """GRPO-compatible trainer with hierarchical token-importance control."""

    _tag_names = ["trl", "tiao", "summarization"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        config = kwargs.get("args")
        if config is None and len(args) >= 3:
            config = args[2]
        if not isinstance(config, TIAOConfig):
            raise TypeError("TIAOTrainer requires TIAOConfig")
        if config.beta != 0.0:
            raise ValueError("TIAO does not use reference-policy KL regularization")

        super().__init__(*args, **kwargs)
        if self.use_liger_loss:
            raise ValueError("TRL 0.19.1 Liger loss does not implement TIAO")
        if not getattr(self.processing_class, "is_fast", False):
            raise ValueError(
                "TIAO source masking requires a fast Qwen tokenizer for "
                "character-to-token offset alignment"
            )

        self._source_mask_token_id, self._source_mask_keeps_attention = (
            self._resolve_source_mask_token()
        )
        if self.accelerator.is_main_process:
            mask_mode = (
                f"visible {_QWEN_FIM_MASK_TOKEN} replacement"
                if self._source_mask_keeps_attention
                else "attention-mask deletion fallback"
            )
            print(
                "TIAO configuration: "
                f"source_mask_probability={_TEXT_MASK_PROBABILITY}, "
                f"token_gate_keep_ratio={_TOKEN_GATE_KEEP_RATIO}, "
                "advantage_scale=trajectory_importance/global_rollout_mean, "
                f"mask_mode={mask_mode}, reference_kl_beta=0, "
                f"loss_type={self.loss_type}",
                flush=True,
            )

    def _resolve_source_mask_token(self) -> tuple[int, bool]:
        """Use Qwen's existing FIM pad token without resizing embeddings."""

        vocabulary = self.processing_class.get_vocab()
        if _QWEN_FIM_MASK_TOKEN in vocabulary:
            return int(vocabulary[_QWEN_FIM_MASK_TOKEN]), True

        mask_token_id = self.processing_class.pad_token_id
        if mask_token_id is None:
            mask_token_id = self.processing_class.eos_token_id
        if mask_token_id is None:
            raise ValueError("No existing token is available for source masking")

        if self.accelerator.is_main_process:
            print(
                f"TIAO warning: {_QWEN_FIM_MASK_TOKEN} is absent; masked source "
                "positions use attention-mask deletion.",
                flush=True,
            )
        return int(mask_token_id), False

    def _build_source_token_mask(
        self,
        inputs: list[dict[str, Any]],
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Align every raw source span to the returned left-padded prompt."""

        source_token_mask = torch.zeros_like(prompt_mask, dtype=torch.bool)
        special_ids = {int(token_id) for token_id in self.processing_class.all_special_ids}

        for row_index, example in enumerate(inputs):
            if "origin_text" not in example:
                raise KeyError("TIAO requires the dataset column 'origin_text'")
            rendered = maybe_apply_chat_template(example, self.processing_class)["prompt"]
            source_text = str(example["origin_text"])
            source_start = rendered.find(source_text)
            if source_start < 0:
                raise ValueError("TIAO could not locate origin_text in the rendered prompt")
            source_end = source_start + len(source_text)

            encoded = self.processing_class(
                rendered,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            encoded_ids = [int(token_id) for token_id in encoded["input_ids"]]
            offsets = [tuple(offset) for offset in encoded["offset_mapping"]]

            active_positions = torch.nonzero(
                prompt_mask[row_index].bool(),
                as_tuple=False,
            ).flatten()
            active_ids = [
                int(token_id)
                for token_id in prompt_ids[
                    row_index, active_positions
                ].detach().cpu().tolist()
            ]
            active_length = len(active_ids)
            if active_length == 0:
                raise ValueError("TIAO received an empty active prompt")

            encoded_start = len(encoded_ids) - active_length
            if encoded_start < 0 or encoded_ids[encoded_start:] != active_ids:
                encoded_start = self._find_token_subsequence(encoded_ids, active_ids)
            if encoded_start < 0:
                raise ValueError(
                    "Prompt-token alignment failed; use the same fast tokenizer "
                    "for rollout and source masking"
                )

            aligned_offsets = offsets[encoded_start : encoded_start + active_length]
            eligible = []
            for token_id, (token_start, token_end) in zip(active_ids, aligned_offsets):
                overlaps_source = token_end > source_start and token_start < source_end
                eligible.append(overlaps_source and token_id not in special_ids)

            eligible_tensor = torch.tensor(
                eligible,
                dtype=torch.bool,
                device=prompt_ids.device,
            )
            source_token_mask[row_index, active_positions] = eligible_tensor
            if not eligible_tensor.any():
                raise ValueError(
                    "No maskable source tokens remain after truncation and alignment"
                )
        return source_token_mask

    @staticmethod
    def _find_token_subsequence(full: list[int], subsequence: list[int]) -> int:
        """Return the final exact occurrence of a subsequence, or -1."""

        if not subsequence or len(subsequence) > len(full):
            return -1
        first = subsequence[0]
        for start in range(len(full) - len(subsequence), -1, -1):
            if full[start] == first and full[start : start + len(subsequence)] == subsequence:
                return start
        return -1

    def _mask_source_tokens(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        source_token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask each trajectory independently while preserving source bounds."""

        random_values = torch.rand(
            source_token_mask.shape,
            dtype=torch.float32,
            device=source_token_mask.device,
        )
        selected = source_token_mask & (random_values < _TEXT_MASK_PROBABILITY)

        for row_index in range(source_token_mask.size(0)):
            candidates = torch.nonzero(
                source_token_mask[row_index],
                as_tuple=False,
            ).flatten()
            selected_count = int(selected[row_index, candidates].sum().item())
            if selected_count == 0:
                chosen = candidates[torch.argmin(random_values[row_index, candidates])]
                selected[row_index, chosen] = True
                selected_count = 1
            if selected_count == candidates.numel() and candidates.numel() > 1:
                visible = candidates[torch.argmax(random_values[row_index, candidates])]
                selected[row_index, visible] = False

        masked_prompt_ids = prompt_ids.clone()
        masked_prompt_ids[selected] = self._source_mask_token_id
        masked_prompt_mask = prompt_mask.clone()
        if not self._source_mask_keeps_attention:
            masked_prompt_mask[selected] = 0
        return masked_prompt_ids, masked_prompt_mask, selected

    @staticmethod
    def _distributed_sum(values: torch.Tensor) -> torch.Tensor:
        reduced = values.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        return reduced

    @staticmethod
    def _distributed_min(value: torch.Tensor) -> torch.Tensor:
        reduced = value.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(reduced, op=dist.ReduceOp.MIN)
        return reduced

    @staticmethod
    def _distributed_max(value: torch.Tensor) -> torch.Tensor:
        reduced = value.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
        return reduced

    def _mean_preserving_advantage_scale(
        self,
        trajectory_importance: torch.Tensor,
    ) -> torch.Tensor:
        """Return score/global_mean(score), with an all-zero identity fallback."""

        statistics = torch.stack(
            [
                trajectory_importance.float().sum(),
                trajectory_importance.new_tensor(
                    trajectory_importance.numel(),
                    dtype=torch.float32,
                ),
            ]
        )
        global_sum, global_count = self._distributed_sum(statistics)
        global_mean = global_sum / global_count.clamp(min=1.0)
        if (
            not torch.isfinite(global_mean)
            or global_mean <= torch.finfo(torch.float32).eps
        ):
            return torch.ones_like(trajectory_importance, dtype=torch.float32)
        return trajectory_importance.float() / global_mean

    def _record_tiao_metrics(
        self,
        mode: str,
        token_importance: torch.Tensor,
        completion_mask: torch.Tensor,
        trajectory_importance: torch.Tensor,
        advantage_scale: torch.Tensor,
        token_gate: torch.Tensor,
        source_token_mask: torch.Tensor,
        selected_source_mask: torch.Tensor,
    ) -> None:
        """Record token-weighted diagnostics reduced over all ranks."""

        valid = completion_mask.bool()
        selected = token_gate & valid
        rejected = valid & ~selected
        sum_statistics = torch.stack(
            [
                (token_importance * valid).sum(),
                valid.sum().to(torch.float32),
                (token_importance * selected).sum(),
                selected.sum().to(torch.float32),
                (token_importance * rejected).sum(),
                rejected.sum().to(torch.float32),
                trajectory_importance.sum(),
                trajectory_importance.new_tensor(
                    trajectory_importance.numel(), dtype=torch.float32
                ),
                advantage_scale.sum(),
                advantage_scale.new_tensor(advantage_scale.numel(), dtype=torch.float32),
                selected_source_mask.sum().to(torch.float32),
                source_token_mask.sum().to(torch.float32),
            ]
        ).float()
        reduced = self._distributed_sum(sum_statistics)

        token_mean = reduced[0] / reduced[1].clamp(min=1.0)
        selected_mean = reduced[2] / reduced[3].clamp(min=1.0)
        rejected_mean = reduced[4] / reduced[5].clamp(min=1.0)
        trajectory_mean = reduced[6] / reduced[7].clamp(min=1.0)
        scale_mean = reduced[8] / reduced[9].clamp(min=1.0)
        source_mask_fraction = reduced[10] / reduced[11].clamp(min=1.0)
        gate_fraction = reduced[3] / reduced[1].clamp(min=1.0)

        trajectory_min = self._distributed_min(trajectory_importance.min())
        trajectory_max = self._distributed_max(trajectory_importance.max())
        scale_min = self._distributed_min(advantage_scale.min())
        scale_max = self._distributed_max(advantage_scale.max())

        metrics = self._metrics[mode]
        metrics["tiao/token_importance_mean"].append(token_mean.item())
        metrics["tiao/selected_token_importance_mean"].append(selected_mean.item())
        metrics["tiao/rejected_token_importance_mean"].append(rejected_mean.item())
        metrics["tiao/trajectory_importance_mean"].append(trajectory_mean.item())
        metrics["tiao/trajectory_importance_min"].append(trajectory_min.item())
        metrics["tiao/trajectory_importance_max"].append(trajectory_max.item())
        metrics["tiao/advantage_scale_mean"].append(scale_mean.item())
        metrics["tiao/advantage_scale_min"].append(scale_min.item())
        metrics["tiao/advantage_scale_max"].append(scale_max.item())
        metrics["tiao/source_mask_fraction"].append(source_mask_fraction.item())
        metrics["tiao/gated_token_fraction"].append(gate_fraction.item())

    def _generate_and_score_completions(
        self,
        inputs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Attach detached TIAO signals to the ordinary rollout batch."""

        batch = super()._generate_and_score_completions(inputs)
        prompt_ids = batch["prompt_ids"]
        prompt_mask = batch["prompt_mask"]
        completion_ids = batch["completion_ids"]
        completion_mask = batch["completion_mask"]
        logits_to_keep = completion_ids.size(1)

        source_token_mask = self._build_source_token_mask(
            inputs,
            prompt_ids,
            prompt_mask,
        )
        masked_prompt_ids, masked_prompt_mask, selected_source_mask = (
            self._mask_source_tokens(prompt_ids, prompt_mask, source_token_mask)
        )

        complete_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        complete_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        masked_input_ids = torch.cat([masked_prompt_ids, completion_ids], dim=1)
        masked_attention_mask = torch.cat(
            [masked_prompt_mask, completion_mask],
            dim=1,
        )
        mode = "train" if self.model.training else "eval"
        forward_batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                complete_logps = self._get_per_token_logps(
                    self.model,
                    complete_input_ids,
                    complete_attention_mask,
                    logits_to_keep,
                    forward_batch_size,
                )
                masked_logps = self._get_per_token_logps(
                    self.model,
                    masked_input_ids,
                    masked_attention_mask,
                    logits_to_keep,
                    forward_batch_size,
                )
        finally:
            self.model.train(was_training)

        token_importance = _low_variance_token_kl(
            complete_logps.detach(),
            masked_logps.detach(),
        )
        valid_counts = completion_mask.sum(dim=1).clamp(min=1).float()
        trajectory_importance = (
            token_importance * completion_mask
        ).sum(dim=1) / valid_counts
        advantage_scale = self._mean_preserving_advantage_scale(
            trajectory_importance.detach()
        )
        shaped_advantages = batch["advantages"].float() * advantage_scale
        token_gate = _exact_top_fraction_mask(token_importance, completion_mask)

        self._record_tiao_metrics(
            mode=mode,
            token_importance=token_importance,
            completion_mask=completion_mask,
            trajectory_importance=trajectory_importance,
            advantage_scale=advantage_scale,
            token_gate=token_gate,
            source_token_mask=source_token_mask,
            selected_source_mask=selected_source_mask,
        )

        batch["advantages"] = shaped_advantages.detach()
        batch["old_per_token_logps"] = complete_logps.detach()
        batch["tiao_token_importance"] = token_importance.detach()
        batch["tiao_trajectory_importance"] = trajectory_importance.detach()
        batch["tiao_advantage_scale"] = advantage_scale.detach()
        batch["tiao_token_gate"] = token_gate.detach()
        return batch

    def _compute_loss(self, model, inputs):
        """Apply the hard token gate after the clipped GRPO surrogate."""

        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        per_token_logps = self._get_per_token_logps(
            model,
            input_ids,
            attention_mask,
            completion_ids.size(1),
        )

        advantages = inputs["advantages"].unsqueeze(1)
        old_per_token_logps = inputs["old_per_token_logps"]
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()

        ratio = torch.exp(per_token_logps - old_per_token_logps)
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.epsilon_low,
            1.0 + self.epsilon_high,
        )
        if self.args.delta is not None:
            ratio = torch.clamp(ratio, max=self.args.delta)

        surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)
        token_gate = inputs["tiao_token_gate"].to(
            device=surrogate.device,
            dtype=surrogate.dtype,
        )
        completion_mask_float = completion_mask.to(surrogate.dtype)
        per_token_loss = -surrogate * token_gate
        masked_loss = per_token_loss * completion_mask_float

        if self.loss_type == "grpo":
            loss = (
                masked_loss.sum(-1)
                / completion_mask_float.sum(-1).clamp(min=1.0)
            ).mean()
        elif self.loss_type == "bnpo":
            loss = masked_loss.sum() / completion_mask_float.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = masked_loss.sum() / (
                per_token_loss.size(0) * self.max_completion_length
            )
        else:
            raise ValueError(f"Unsupported TIAO loss_type: {self.loss_type}")

        mode = "train" if self.model.training else "eval"
        positive_advantage = advantages > 0
        low_clipped = (ratio < 1.0 - self.epsilon_low) & ~positive_advantage
        high_clipped = (ratio > 1.0 + self.epsilon_high) & positive_advantage
        region_clipped = low_clipped | high_clipped
        valid_tokens = completion_mask_float.sum().clamp(min=1.0)
        gated_valid = completion_mask_float * token_gate

        low_ratio = (low_clipped * completion_mask_float).sum() / valid_tokens
        high_ratio = (high_clipped * completion_mask_float).sum() / valid_tokens
        clip_ratio = (region_clipped * completion_mask_float).sum() / valid_tokens
        gated_clip_ratio = (
            region_clipped * gated_valid
        ).sum() / gated_valid.sum().clamp(min=1.0)

        gathered_low = self.accelerator.gather(low_ratio.reshape(1))
        gathered_high = self.accelerator.gather(high_ratio.reshape(1))
        gathered_clip = self.accelerator.gather(clip_ratio.reshape(1))
        gathered_gated_clip = self.accelerator.gather(gated_clip_ratio.reshape(1))
        self._metrics[mode]["clip_ratio/low_mean"].append(
            gathered_low.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/low_min"].append(gathered_low.min().item())
        self._metrics[mode]["clip_ratio/high_mean"].append(
            gathered_high.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/high_max"].append(gathered_high.max().item())
        self._metrics[mode]["clip_ratio/region_mean"].append(
            gathered_clip.nanmean().item()
        )
        self._metrics[mode]["tiao/gated_clip_ratio"].append(
            gathered_gated_clip.nanmean().item()
        )
        return loss
