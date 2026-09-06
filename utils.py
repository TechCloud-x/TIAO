"""Utilities for constructing and reporting UniEval inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from prettytable import PrettyTable


def convert_to_json(
    output_list: Sequence[str],
    src_list: Sequence[str] | None = None,
    ref_list: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """Convert aligned summaries, sources, and references to UniEval records."""

    size = len(output_list)
    if src_list is not None and len(src_list) != size:
        raise ValueError("src_list must have the same length as output_list")
    if ref_list is not None and len(ref_list) != size:
        raise ValueError("ref_list must have the same length as output_list")

    records: list[dict[str, str]] = []
    for index, output in enumerate(output_list):
        record = {"system_output": str(output)}
        if src_list is not None:
            record["source"] = str(src_list[index])
        if ref_list is not None:
            record["reference"] = str(ref_list[index])
        records.append(record)
    return records


def add_question(
    dimension: str,
    output: Sequence[str],
    src: Sequence[str] | None = None,
    ref: Sequence[str] | None = None,
    task: str = "summarization",
) -> list[str]:
    """Format a supported summarization dimension as a Boolean QA prompt."""

    if task != "summarization":
        raise NotImplementedError("Only summarization is supported in this release")

    sources = src if src is not None else [""] * len(output)
    references = ref if ref is not None else [""] * len(output)
    if len(sources) != len(output) or len(references) != len(output):
        raise ValueError("output, src, and ref must be aligned")

    prompts: list[str] = []
    for index, generated_text in enumerate(output):
        if dimension == "fluency":
            prompt = f"question: Is this a fluent paragraph? </s> paragraph: {generated_text}"
        elif dimension == "coherence":
            prompt = (
                "question: Is this a coherent summary to the document? </s> "
                f"summary: {generated_text} </s> document: {sources[index]}"
            )
        elif dimension == "consistency":
            prompt = (
                "question: Is this claim consistent with the document? </s> "
                f"claim: {generated_text} </s> document: {sources[index]}"
            )
        elif dimension == "relevance":
            prompt = (
                "question: Is this summary relevant to the reference? </s> "
                f"summary: {generated_text} </s> reference: {references[index]}"
            )
        else:
            raise NotImplementedError(f"Unsupported UniEval dimension: {dimension}")
        prompts.append(prompt)
    return prompts


def print_scores(scores: Sequence[dict[str, Any]]) -> None:
    """Print macro averages for a non-empty sequence of score dictionaries."""

    if not scores:
        raise ValueError("scores must not be empty")
    table = PrettyTable(["Dimension", "Score"])
    for dimension in scores[0]:
        mean_score = sum(float(item[dimension]) for item in scores) / len(scores)
        table.add_row([dimension, round(mean_score, 6)])
    print("\nEvaluation scores:")
    print(table)

