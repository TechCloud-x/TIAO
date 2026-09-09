"""TIAO source-mask probability ablations for text summarization.

This module changes only the probability used to mask eligible source tokens.
The trajectory scaling, exact top-40% token gate, clipped policy surrogate,
and zero reference-policy KL behavior are inherited unchanged from TIAO.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from tiao_rollout_trainer import TIAORolloutTrainer
from tiao_trainer import TIAOConfig, TIAOTrainer, _QWEN_FIM_MASK_TOKEN


@dataclass
class TIAOMaskAblationConfig(TIAOConfig):
    """TIAO configuration with a validated source-mask probability."""

    source_mask_probability: float = field(
        default=0.5,
        metadata={
            "help": (
                "Independent probability of masking each eligible source token; "
                "must satisfy 0 < source_mask_probability < 1."
            )
        },
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        probability = float(self.source_mask_probability)
        if not math.isfinite(probability) or not 0.0 < probability < 1.0:
            raise ValueError(
                "--source_mask_probability must satisfy "
                "0 < source_mask_probability < 1"
            )
        self.source_mask_probability = probability


class TIAOMaskAblationTrainer(TIAOTrainer):
    """TIAO trainer whose only algorithmic variable is source masking."""

    _tag_names = ["trl", "tiao", "summarization", "source-mask-ablation"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        config = kwargs.get("args")
        if config is None and len(args) >= 3:
            config = args[2]
        if not isinstance(config, TIAOMaskAblationConfig):
            raise TypeError(
                "TIAOMaskAblationTrainer requires TIAOMaskAblationConfig"
            )
        if config.beta != 0.0:
            raise ValueError("TIAO does not use reference-policy KL regularization")

        # Initialize the shared rollout layer directly because the baseline TIAO
        # constructor reports its fixed 0.5 probability. All trainer methods not
        # overridden below remain inherited from the baseline implementation.
        TIAORolloutTrainer.__init__(self, *args, **kwargs)
        if self.use_liger_loss:
            raise ValueError("TRL 0.19.1 Liger loss does not implement TIAO")
        if not getattr(self.processing_class, "is_fast", False):
            raise ValueError(
                "TIAO source masking requires a fast Qwen tokenizer for "
                "character-to-token offset alignment"
            )

        self.source_mask_probability = config.source_mask_probability
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
                "TIAO source-mask ablation configuration: "
                f"source_mask_probability={self.source_mask_probability}, "
                "token_gate_keep_ratio=0.4, "
                "advantage_scale=trajectory_importance/global_rollout_mean, "
                f"mask_mode={mask_mode}, reference_kl_beta=0, "
                f"loss_type={self.loss_type}",
                flush=True,
            )

    def _mask_source_tokens(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        source_token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask source tokens independently at the configured probability."""

        random_values = torch.rand(
            source_token_mask.shape,
            dtype=torch.float32,
            device=source_token_mask.device,
        )
        selected = source_token_mask & (
            random_values < self.source_mask_probability
        )

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
