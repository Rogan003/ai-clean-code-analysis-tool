#!/usr/bin/env python3
"""
Load and process a code readability dataset from Hugging Face.

This script focuses on the dataset referenced in the README:
  - https://huggingface.co/datasets/se2p/code-readability-merged

It will:
  1) Download the dataset using the `datasets` library.
  2) Detect whether each code snippet looks like a Java class (class/interface/enum)
     or a Java method (method declaration/definition) using lightweight regex heuristics.
  3) Keep only samples that are either a class or a method.
  4) Map the numeric readability score to 3 bins:
        - good                (>= 4.0)
        - changes_recommended ([3.0, 4.0))
        - changes_required    (< 3.0)
  5) Save the processed dataset to CSV with three columns:
        - code_snippet (only full classes or methods)
        - code_type (class or method)
        - code_score (good, changes_recommended, or changes_required)

Usage:
  python download_and_parse_hf_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterable

from datasets import load_dataset
from tqdm import tqdm

# ----------------------------
# Heuristic detection helpers
# ----------------------------

_CLASS_DECL_RE = re.compile(r"\b(class|interface|enum)\s+[A-Za-z_$][A-Za-z0-9_$]*\b")

# Rough Java method signature detector that avoids matching method calls.
# It expects a return type (or 'void') before the name and parameter list, and a body '{' or ';'.
_METHOD_DECL_RE = re.compile(
    r"(?m)^\s*"
    r"(?:(?:public|protected|private|static|final|native|synchronized|abstract|transient|strictfp|default)\s+)*"  # modifiers
    r"(?:<[^>\n]+>\s*)?"  # optional generics before return type
    r"(?:void|[A-Za-z_$][A-Za-z0-9_$]*(?:\s*<[^>]+>)?(?:\[\s*\])*)\s+"  # return type or void
    r"[A-Za-z_$][A-Za-z0-9_$]*\s*"  # method name
    r"\([^;{}]*\)\s*"  # parameters
    r"(?:throws\s+[A-Za-z0-9_$.,\s]+\s*)?"  # optional throws
    r"[;{]"  # abstract/interface method or method with body
)

_COMMENT_BLOCK_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_COMMENT_LINE_RE = re.compile(r"//.*?$", re.MULTILINE)


def strip_comments(code: str) -> str:
    code = _COMMENT_BLOCK_RE.sub(" ", code)
    code = _COMMENT_LINE_RE.sub(" ", code)
    return code


def classify_snippet_type(java_code: str) -> Optional[str]:
    """Classify a snippet as 'class' or 'method' using regex heuristics.

    Returns:
        'class' | 'method' | None
    """
    if not java_code or not isinstance(java_code, str):
        return None

    text = strip_comments(java_code)

    # Quick disqualifiers to reduce false positives in messy fragments
    if "(" not in text and "class" not in text and "interface" not in text and "enum" not in text:
        return None

    # Class / interface / enum detection
    if _CLASS_DECL_RE.search(text):
        return "class"

    # Method detection (rough but effective for typical Java signatures)
    if _METHOD_DECL_RE.search(text):
        return "method"

    return None


# ----------------------------
# Score binning
# ----------------------------

def map_score_to_label(score: float) -> str:
    """Map continuous score to categorical label.

    - good: >= 4.0
    - changes_recommended: [3.0, 4.0)
    - changes_required: < 3.0
    """
    try:
        s = float(score)
    except Exception:
        # If score is missing or unparsable, default to the most conservative label
        return "changes_required"

    if s >= 4.0:
        return "good"
    if s >= 3.0:
        return "changes_recommended"
    return "changes_required"


# ----------------------------
# Dataset loading and processing
# ----------------------------

@dataclass
class ProcessConfig:
    dataset_name: str
    split: str
    save_path: str
    out_format: str  # 'csv' or 'json'
    max_samples: Optional[int]
    print_sample: int


def resolve_columns(example: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize varying dataset column names to a standard contract.

    Expected source columns in HF datasets (observed):
      - code_snippet: str
      - score: float
      - name: str (optional)

    Some viewers show unnamed columns, but the underlying records typically use
    `code_snippet` and `score`. This function attempts to be permissive.
    """
    code = (
        example.get("code_snippet")
        or example.get("code")
        or example.get("text")
        or example.get("snippet")
    )
    score = example.get("score")
    name = example.get("name") or example.get("id") or ""
    return {"code_snippet": code, "score": score, "name": name}


def process_dataset(cfg: ProcessConfig) -> List[Dict[str, Any]]:
    ds = load_dataset(cfg.dataset_name, split=cfg.split)

    processed: List[Dict[str, Any]] = []

    total = len(ds)
    limit = min(cfg.max_samples, total) if cfg.max_samples is not None else total

    for ex in tqdm(ds.select(range(limit)), desc=f"Processing {cfg.dataset_name}"):
        row = resolve_columns(ex)
        code = row["code_snippet"]
        score = row["score"]

        stype = classify_snippet_type(code)
        if stype is None:
            continue  # keep only class or method

        label = map_score_to_label(score)

        processed.append(
            {
                "code_snippet": code,
                "score": float(score) if score is not None else None,
                "readability_label": label,
                "snippet_type": stype,
                "source_dataset": cfg.dataset_name,
            }
        )

    return processed


def save_output(rows: List[Dict[str, Any]], path: str, fmt: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fmt == "csv":
        if not rows:
            # Save header only for consistency
            header = [
                "code_snippet",
                "score",
                "readability_label",
                "snippet_type",
                "name",
                "source_dataset",
            ]
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
            return

        header = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    elif fmt == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def print_summary(rows: List[Dict[str, Any]], total_before: int) -> None:
    kept = len(rows)
    print("")
    print("=== Summary ===")
    print(f"Total rows before filtering: {total_before}")
    print(f"Kept (class/method only):    {kept}")

    by_type: Dict[str, int] = {}
    by_label: Dict[str, int] = {}
    for r in rows:
        by_type[r["snippet_type"]] = by_type.get(r["snippet_type"], 0) + 1
        by_label[r["readability_label"]] = by_label.get(r["readability_label"], 0) + 1

    print("- By snippet_type:")
    for k in sorted(by_type.keys()):
        print(f"  {k:>7}: {by_type[k]}")

    print("- By readability_label:")
    for k in ("good", "changes_recommended", "changes_required"):
        if k in by_label:
            print(f"  {k:>20}: {by_label[k]}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Download and process code readability datasets.")
    parser.add_argument(
        "--dataset",
        default="se2p/code-readability-merged",
        help="Hugging Face dataset name (default: se2p/code-readability-merged)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--save-path",
        default=os.path.join("data", "processed", "merged_processed.csv"),
        help="Output file path (default: data/processed/merged_processed.csv)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format (csv or json). Default: csv",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of samples processed (for quick exploration).",
    )
    parser.add_argument(
        "--print-sample",
        type=int,
        default=0,
        help="Print N sample rows to stdout after processing.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    ds = load_dataset(args.dataset, split=args.split)
    total_before = len(ds if args.max_samples is None else ds.select(range(min(len(ds), args.max_samples))))

    cfg = ProcessConfig(
        dataset_name=args.dataset,
        split=args.split,
        save_path=args.save_path,
        out_format=args.format,
        max_samples=args.max_samples,
        print_sample=args.print_sample,
    )

    rows = process_dataset(cfg)

    # Save
    save_output(rows, args.save_path, args.format)

    # Report
    print_summary(rows, total_before)

    # Optionally print a few samples
    if args.print_sample and rows:
        print("")
        print("=== Sample rows ===")
        for r in rows[: max(0, args.print_sample)]:
            print(json.dumps(r, ensure_ascii=False) if args.format == "json" else r)


if __name__ == "__main__":
    main()
