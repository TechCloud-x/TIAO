"""UniEval scoring for abstractive summarization."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import deepspeed
import numpy as np
import torch
from nltk import sent_tokenize
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from utils import add_question, print_scores


class UniEvaluator:
    """Score Boolean QA prompts with a UniEval sequence-to-sequence model."""

    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 1024,
        device: str = "cuda:0",
        cache_dir: str | None = None,
        deepspeed_config: str | None = None,
        world_size: int | None = None,
        show_progress: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.max_length = max_length
        self.show_progress = show_progress
        self.inference_dtype = (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )
        local_files_only = Path(model_name_or_path).exists()
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            config=self.config,
            cache_dir=cache_dir,
            torch_dtype=self.inference_dtype,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
            use_safetensors=True,
        )
        self.model.eval()

        if deepspeed_config:
            with open(deepspeed_config, "r", encoding="utf-8") as config_file:
                inference_config = json.load(config_file)
            if world_size is not None and "tensor_parallel" in inference_config:
                inference_config["tensor_parallel"]["tp_size"] = world_size
            self.model = deepspeed.init_inference(self.model, config=inference_config)
        else:
            self.model.to(device=self.device, dtype=self.inference_dtype)

        self.positive_id = self.tokenizer("Yes")["input_ids"][0]
        self.negative_id = self.tokenizer("No")["input_ids"][0]

    def _model_device(self) -> torch.device:
        model = getattr(self.model, "module", self.model)
        return next(model.parameters()).device

    def score(self, inputs: list[str], batch_size: int = 8) -> list[float]:
        """Return P(Yes) normalized over the Yes and No token logits."""

        if not inputs:
            return []
        targets = ["No"] * len(inputs)
        scores: list[float] = []
        for start in tqdm(
            range(0, len(inputs), batch_size),
            disable=not self.show_progress,
        ):
            source_batch = inputs[start : start + batch_size]
            target_batch = targets[start : start + batch_size]
            try:
                with torch.inference_mode():
                    encoded_source = self.tokenizer(
                        source_batch,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    encoded_target = self.tokenizer(
                        target_batch,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    device = self._model_device()
                    target_ids = encoded_target["input_ids"].to(device)[:, :1]
                    output = self.model(
                        input_ids=encoded_source["input_ids"].to(device),
                        attention_mask=encoded_source["attention_mask"].to(device),
                        labels=target_ids,
                    )
                    logits = output.logits.reshape(-1, self.config.vocab_size)
                    yes_no = torch.softmax(
                        logits[:, [self.positive_id, self.negative_id]], dim=1
                    )
                    scores.extend(yes_no[:, 0].float().cpu().tolist())
            except RuntimeError as exc:
                raise RuntimeError(
                    "UniEval inference failed at batch offset "
                    f"{start} with batch size {len(source_batch)}"
                ) from exc
        return scores


def _sentences(text: str) -> list[str]:
    """Split sentences with NLTK and retain a deterministic fallback."""

    try:
        return sent_tokenize(text)
    except LookupError:
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


class SumEvaluator:
    """Evaluate coherence, consistency, fluency, and relevance."""

    dimensions = ("coherence", "consistency", "fluency", "relevance")

    def __init__(
        self,
        max_length: int = 1024,
        device: str = "cuda:0",
        cache_dir: str | None = None,
        deepspeed_config: str | None = None,
        unieval_model_name_or_path: str | None = None,
        world_size: int | None = None,
        show_progress: bool = True,
    ) -> None:
        if not unieval_model_name_or_path:
            raise ValueError("unieval_model_name_or_path is required")
        self.scorer = UniEvaluator(
            model_name_or_path=unieval_model_name_or_path,
            max_length=max_length,
            device=device,
            cache_dir=cache_dir,
            deepspeed_config=deepspeed_config,
            world_size=world_size,
            show_progress=show_progress,
        )
        self.show_progress = show_progress

    def evaluate(
        self,
        data: list[dict[str, str]],
        dims: list[str] | None = None,
        overall: bool = True,
        print_result: bool = False,
    ) -> list[dict[str, float]]:
        """Evaluate the requested dimensions for every aligned record."""

        eval_dimensions = list(self.dimensions if dims is None else dims)
        unsupported = set(eval_dimensions).difference(self.dimensions)
        if unsupported:
            raise NotImplementedError(f"Unsupported dimensions: {sorted(unsupported)}")

        results: list[dict[str, float]] = [{} for _ in data]
        for dimension in eval_dimensions:
            if self.show_progress:
                print(f"Evaluating {dimension} for {len(data)} samples")

            if dimension in {"consistency", "fluency"}:
                question_outputs: list[str] = []
                question_sources: list[str] = []
                sentence_counts: list[int] = []
                for record in data:
                    sentences = _sentences(record["system_output"])
                    sentence_counts.append(len(sentences))
                    question_outputs.extend(sentences)
                    question_sources.extend(
                        [record["source"] if dimension == "consistency" else ""]
                        * len(sentences)
                    )
                questions = add_question(
                    dimension=dimension,
                    output=question_outputs,
                    src=question_sources,
                )
                sentence_scores = self.scorer.score(questions)
                scores: list[float] = []
                offset = 0
                for count in sentence_counts:
                    if count:
                        scores.append(float(np.mean(sentence_scores[offset : offset + count])))
                    else:
                        scores.append(0.0)
                    offset += count
            else:
                outputs = [record["system_output"] for record in data]
                sources = [record["source"] for record in data]
                references = [record.get("reference", "") for record in data]
                questions = add_question(
                    dimension=dimension,
                    output=outputs,
                    src=sources,
                    ref=references,
                )
                scores = self.scorer.score(questions)

            for index, score in enumerate(scores):
                results[index][dimension] = float(score)

        if overall:
            for result in results:
                result["overall"] = float(np.mean(list(result.values())))
        if print_result:
            print_scores(results)
        return results

