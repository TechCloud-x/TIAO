#!/usr/bin/env python3
"""Distributed inference and sample-weighted evaluation on CNN/DailyMail test."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from unieval import SumEvaluator
from utils import convert_to_json


UNIEVAL_DIMENSIONS = ("coherence", "consistency", "fluency", "relevance")
AGGREGATE_KEYS = (*UNIEVAL_DIMENSIONS, "overall", "sum")


def parse_sample_count(value: str) -> int | None:
    """Parse a positive sample count or the complete-split sentinel."""

    if value.lower() == "all":
        return None
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "test-samples must be a positive integer or 'all'"
        ) from exc
    if count <= 0:
        raise argparse.ArgumentTypeError("test-samples must be a positive integer or 'all'")
    return count


def parse_args() -> argparse.Namespace:
    """Parse inference and evaluation options."""

    repository_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Trained model or checkpoint directory supplied by the launcher.",
    )
    parser.add_argument(
        "--unieval-model-path",
        type=Path,
        default=repository_root / "models" / "unieval-sum",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=repository_root / "data" / "cnn_dailymail" / "3.0.0",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--test-samples", type=parse_sample_count, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--unieval-max-length", type=int, default=1024)
    parser.add_argument(
        "--skip-unieval",
        action="store_true",
        help="Generate predictions without calculating UniEval metrics.",
    )
    args = parser.parse_args()
    for argument_name in (
        "generation_batch_size",
        "evaluation_batch_size",
        "max_prompt_length",
        "max_new_tokens",
        "unieval_max_length",
    ):
        if getattr(args, argument_name) <= 0:
            parser.error(f"--{argument_name.replace('_', '-')} must be positive")
    if args.model_label is None:
        args.model_label = args.model_path.resolve().name
    return args


def build_prompt(text: str) -> str:
    """Construct the same summarization prompt used during training."""

    return (
        f"Text: {text}\n"
        "Instruction: Summarize the Text without any Explanation.\n"
        "Output:"
    )


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON file."""

    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Atomically replace a JSON Lines file."""

    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    os.replace(temporary_path, path)


def setup_distributed() -> tuple[int, int, int, torch.device]:
    """Initialize one NCCL process per GPU."""

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("CNN/DailyMail inference requires CUDA GPUs")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=12))
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def synchronize(world_size: int) -> None:
    """Synchronize ranks when distributed execution is active."""

    if world_size > 1:
        dist.barrier()


def load_test_dataset(dataset_path: Path, sample_count: int | None, seed: int):
    """Load the complete test split, or a deterministic diagnostic subset."""

    dataset = load_dataset(
        "parquet",
        data_files=str(dataset_path / "test-*.parquet"),
        split="train",
    )
    if sample_count is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(sample_count, len(dataset))))
    return dataset


def unique_bigram_ratio(text: str) -> float:
    """Return the fraction of unique whitespace-token bigrams."""

    words = text.split()
    if len(words) < 2:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    return len(set(bigrams)) / len(bigrams)


def generate_records(
    args: argparse.Namespace,
    dataset,
    positions: list[int],
    device: torch.device,
    rank: int,
) -> tuple[list[dict[str, Any]], float]:
    """Generate summaries for this rank's non-overlapping strided shard."""

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    model.config.use_cache = True

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for offset in range(0, len(positions), args.generation_batch_size):
        batch_positions = positions[offset : offset + args.generation_batch_size]
        rows = [dataset[position] for position in batch_positions]
        prompts = [build_prompt(row["article"]) for row in rows]
        chat_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        encoded = tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
            add_special_tokens=False,
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion_ids = generated[:, encoded["input_ids"].shape[1] :]
        summaries = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        for position, row, summary, token_ids in zip(
            batch_positions, rows, summaries, completion_ids
        ):
            summary = summary.strip()
            generated_tokens = int((token_ids != tokenizer.pad_token_id).sum().item())
            source_words = len(row["article"].split())
            prediction_words = len(summary.split())
            records.append(
                {
                    "sample_order": position,
                    "sample_id": hashlib.sha256(
                        row["article"].encode("utf-8")
                    ).hexdigest()[:16],
                    "rank": rank,
                    "source": row["article"],
                    "reference": row["highlights"],
                    "prediction": summary,
                    "generated_tokens": generated_tokens,
                    "reached_max_new_tokens": generated_tokens >= args.max_new_tokens,
                    "source_words": source_words,
                    "prediction_words": prediction_words,
                    "compression_ratio": (
                        source_words / prediction_words if prediction_words else None
                    ),
                    "repetition_score": unique_bigram_ratio(summary),
                }
            )
        if rank == 0:
            completed = min(offset + len(batch_positions), len(positions))
            print(f"Generation progress on rank 0: {completed}/{len(positions)}", flush=True)

    elapsed = time.perf_counter() - started
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return records, elapsed


def empty_accumulator() -> dict[str, dict[str, float | int | None]]:
    """Create a sample-level statistics accumulator."""

    return {
        key: {"sum": 0.0, "sum_squares": 0.0, "count": 0, "min": None, "max": None}
        for key in AGGREGATE_KEYS
    }


def add_value(
    accumulator: dict[str, dict[str, float | int | None]], key: str, value: float
) -> None:
    """Add one sample score without introducing a batch-level mean."""

    if not math.isfinite(value):
        return
    item = accumulator[key]
    item["sum"] = float(item["sum"]) + value
    item["sum_squares"] = float(item["sum_squares"]) + value * value
    item["count"] = int(item["count"]) + 1
    item["min"] = value if item["min"] is None else min(float(item["min"]), value)
    item["max"] = value if item["max"] is None else max(float(item["max"]), value)


def evaluate_records(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[float, dict[str, dict[str, float | int | None]]]:
    """Score local records and retain one value per sample and dimension."""

    accumulator = empty_accumulator()
    if args.skip_unieval or not records:
        return 0.0, accumulator

    started = time.perf_counter()
    evaluator = SumEvaluator(
        max_length=args.unieval_max_length,
        cache_dir=None,
        device=str(device),
        deepspeed_config=None,
        unieval_model_name_or_path=str(args.unieval_model_path),
        world_size=world_size,
        show_progress=False,
    )
    for offset in range(0, len(records), args.evaluation_batch_size):
        batch_records = records[offset : offset + args.evaluation_batch_size]
        evaluation_data = convert_to_json(
            output_list=[record["prediction"] for record in batch_records],
            src_list=[record["source"] for record in batch_records],
            ref_list=[record["reference"] for record in batch_records],
        )
        scores = evaluator.evaluate(
            evaluation_data,
            dims=list(UNIEVAL_DIMENSIONS),
            overall=True,
        )
        for record, score in zip(batch_records, scores):
            per_sample = {
                name: float(score[name]) for name in (*UNIEVAL_DIMENSIONS, "overall")
            }
            per_sample["sum"] = sum(per_sample[name] for name in UNIEVAL_DIMENSIONS)
            record["unieval"] = per_sample
            for name, value in per_sample.items():
                add_value(accumulator, name, value)
        if rank == 0:
            completed = min(offset + len(batch_records), len(records))
            print(f"UniEval progress on rank 0: {completed}/{len(records)}", flush=True)

    elapsed = time.perf_counter() - started
    del evaluator
    gc.collect()
    torch.cuda.empty_cache()
    return elapsed, accumulator


def merge_accumulators(
    accumulators: Iterable[dict[str, dict[str, float | int | None]]],
) -> dict[str, dict[str, float | int | None]]:
    """Merge rank-local sample accumulators without averaging rank means."""

    merged = empty_accumulator()
    for accumulator in accumulators:
        for key in AGGREGATE_KEYS:
            source = accumulator[key]
            target = merged[key]
            target["sum"] = float(target["sum"]) + float(source["sum"])
            target["sum_squares"] = float(target["sum_squares"]) + float(
                source["sum_squares"]
            )
            target["count"] = int(target["count"]) + int(source["count"])
            if source["min"] is not None:
                target["min"] = (
                    float(source["min"])
                    if target["min"] is None
                    else min(float(target["min"]), float(source["min"]))
                )
            if source["max"] is not None:
                target["max"] = (
                    float(source["max"])
                    if target["max"] is None
                    else max(float(target["max"]), float(source["max"]))
                )
    return merged


def finalize_accumulator(
    accumulator: dict[str, dict[str, float | int | None]],
) -> dict[str, dict[str, float | int | None]]:
    """Compute globally sample-weighted descriptive statistics."""

    metrics: dict[str, dict[str, float | int | None]] = {}
    for key, item in accumulator.items():
        count = int(item["count"])
        if count == 0:
            metrics[key] = {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
            continue
        total = float(item["sum"])
        mean = total / count
        variance = max(float(item["sum_squares"]) / count - mean * mean, 0.0)
        metrics[key] = {
            "count": count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": item["min"],
            "max": item["max"],
        }
    return metrics


def numeric_summary(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Compute sample-weighted statistics for a numeric record field."""

    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(values) / len(values)
    variance = max(sum(value * value for value in values) / len(values) - mean * mean, 0.0)
    return {
        "count": len(values),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def merge_rank_outputs(args: argparse.Namespace, world_size: int, parts_dir: Path) -> None:
    """Merge rank shards and calculate final metrics over all test samples."""

    records: list[dict[str, Any]] = []
    rank_metadata: list[dict[str, Any]] = []
    for rank in range(world_size):
        part_path = parts_dir / f"rank-{rank:05d}.jsonl"
        metadata_path = parts_dir / f"rank-{rank:05d}.meta.json"
        if not part_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing rank output for rank {rank}: {part_path} / {metadata_path}"
            )
        with part_path.open("r", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
        with metadata_path.open("r", encoding="utf-8") as handle:
            rank_metadata.append(json.load(handle))

    records.sort(key=lambda record: record["sample_order"])
    expected_counts = {item["selected_sample_count"] for item in rank_metadata}
    if len(expected_counts) != 1:
        raise RuntimeError(f"Ranks disagree on selected test size: {sorted(expected_counts)}")
    expected_count = expected_counts.pop()
    if len(records) != expected_count:
        raise RuntimeError(f"Merged {len(records)} predictions, expected {expected_count}")
    if [record["sample_order"] for record in records] != list(range(expected_count)):
        raise RuntimeError("Merged predictions contain missing or duplicate sample indices")

    unieval_metrics: dict[str, Any] = {}
    if not args.skip_unieval:
        merged = merge_accumulators(item["unieval_accumulator"] for item in rank_metadata)
        unieval_metrics = finalize_accumulator(merged)
        for key in AGGREGATE_KEYS:
            if unieval_metrics[key]["count"] != expected_count:
                raise RuntimeError(
                    f"UniEval metric {key} has {unieval_metrics[key]['count']} samples; "
                    f"expected {expected_count}"
                )

    generation_seconds = max(
        (item["generation_seconds"] for item in rank_metadata), default=0.0
    )
    evaluation_seconds = max(
        (item["evaluation_seconds"] for item in rank_metadata), default=0.0
    )
    metrics = {
        "model_label": args.model_label,
        "model_path": str(args.model_path.resolve()),
        "dataset_path": str(args.dataset_path.resolve()),
        "split": "test",
        "sample_count": len(records),
        "sample_seed": args.seed if args.test_samples is not None else None,
        "world_size": world_size,
        "averaging_method": (
            "sum every sample score globally, then divide by the global sample count"
        ),
        "decoding": {
            "strategy": "greedy",
            "generation_batch_size_per_rank": args.generation_batch_size,
            "evaluation_batch_size_per_rank": args.evaluation_batch_size,
            "max_prompt_length": args.max_prompt_length,
            "max_new_tokens": args.max_new_tokens,
        },
        "timing": {
            "generation_wall_seconds": generation_seconds,
            "evaluation_wall_seconds": evaluation_seconds,
            "generation_samples_per_second": (
                len(records) / generation_seconds if generation_seconds else None
            ),
        },
        "prediction": {
            "generated_tokens": numeric_summary(records, "generated_tokens"),
            "prediction_words": numeric_summary(records, "prediction_words"),
            "compression_ratio": numeric_summary(records, "compression_ratio"),
            "repetition_score": numeric_summary(records, "repetition_score"),
            "empty_output_rate": (
                sum(not record["prediction"] for record in records) / len(records)
                if records
                else None
            ),
            "reached_max_new_tokens_rate": (
                sum(record["reached_max_new_tokens"] for record in records) / len(records)
                if records
                else None
            ),
        },
        "unieval": unieval_metrics,
        "rank_metadata": rank_metadata,
    }
    atomic_write_jsonl(args.output_dir / "predictions.jsonl", records)
    atomic_write_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    """Run distributed test inference and global sample-level aggregation."""

    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()
    try:
        set_seed(args.seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        for path_name, path in (
            ("model", args.model_path),
            ("dataset", args.dataset_path),
            ("UniEval model", args.unieval_model_path),
        ):
            if path_name == "UniEval model" and args.skip_unieval:
                continue
            if not path.exists():
                raise FileNotFoundError(f"{path_name} path does not exist: {path}")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        parts_dir = args.output_dir / "parts"
        parts_dir.mkdir(parents=True, exist_ok=True)

        dataset = None
        if rank == 0:
            dataset = load_test_dataset(args.dataset_path, args.test_samples, args.seed)
            print(f"Selected {len(dataset)} samples from CNN/DailyMail test", flush=True)
        synchronize(world_size)
        if rank != 0:
            dataset = load_test_dataset(args.dataset_path, args.test_samples, args.seed)
        synchronize(world_size)
        assert dataset is not None

        positions = list(range(rank, len(dataset), world_size))
        records, generation_seconds = generate_records(
            args, dataset, positions, device, rank
        )
        synchronize(world_size)
        evaluation_seconds, accumulator = evaluate_records(
            args, records, device, rank, world_size
        )

        atomic_write_jsonl(parts_dir / f"rank-{rank:05d}.jsonl", records)
        atomic_write_json(
            parts_dir / f"rank-{rank:05d}.meta.json",
            {
                "rank": rank,
                "local_rank": local_rank,
                "sample_count": len(records),
                "selected_sample_count": len(dataset),
                "generation_seconds": generation_seconds,
                "evaluation_seconds": evaluation_seconds,
                "unieval_accumulator": accumulator,
            },
        )
        synchronize(world_size)
        if rank == 0:
            merge_rank_outputs(args, world_size, parts_dir)
        synchronize(world_size)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
