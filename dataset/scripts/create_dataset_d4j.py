#!/usr/bin/env python3
"""
Create a CSV dataset from Java files in the D4J dump.

This script processes Java files and uses lightweight regex heuristics to classify
code snippets as classes or methods, similar to download_and_parse_hf_dataset.py.

Outputs a CSV with columns:
- code_snippet: the source text
- score: numeric readability score (left empty for now)
- readability_label: categorical label ("good", "changes_recommended", "changes_required")
- snippet_type: "class" or "method"
- name: optional identifier
- source_dataset: dataset identifier

Usage:
  python dataset/scripts/create_dataset_d4j.py \
      --input-dir dataset/data/classes/85K_Files_Dataset_D4J \
      --save-path dataset/data/processed/d4j_processed.csv \
      --min-rows 20001
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Optional

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x

# ----------------------------
# Heuristic detection helpers (matching download_and_parse_hf_dataset.py)
# ----------------------------

_CLASS_DECL_RE = re.compile(r"\b(class|interface|enum)\s+[A-Za-z_$][A-Za-z0-9_$]*\b")

# Rough Java method signature detector that avoids matching method calls.
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
    input_dir: str
    save_path: str
    out_format: str  # 'csv' or 'json'
    min_rows: int
    max_files: Optional[int]
    source_dataset: str = "D4J"


def iter_java_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith('.java'):
                yield os.path.join(dirpath, fn)


def process_dataset(cfg: ProcessConfig) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []
    files = list(iter_java_files(cfg.input_dir))
    if cfg.max_files is not None:
        files = files[:cfg.max_files]

    for path in tqdm(files, desc='Processing Java files'):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                src = f.read()
        except Exception:
            continue

        # Classify the entire file content
        stype = classify_snippet_type(src)
        if stype is None:
            continue  # keep only class or method

        # Since we don't have scores for D4J, leave them as None
        # and use default label
        label = "changes_required"  # default label for unscored data

        processed.append({
            "code_snippet": src,
            "score": None,
            "readability_label": label,
            "snippet_type": stype,
            "name": os.path.basename(path),
            "source_dataset": cfg.source_dataset,
        })

        if len(processed) > cfg.min_rows:
            break

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
    print(f"Total files processed: {total_before}")
    print(f"Kept (class/method only): {kept}")

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


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Create D4J dataset with HF-compatible format.')
    p.add_argument('--input-dir', default='dataset/data/classes/85K_Files_Dataset_D4J', help='Directory containing .java files')
    p.add_argument('--save-path', default='dataset/data/processed/d4j_processed.csv', help='Output file path')
    p.add_argument('--format', choices=['csv', 'json'], default='csv', help='Output format (csv or json). Default: csv')
    p.add_argument('--min-rows', type=int, default=20001, help='Stop when strictly more than this number of rows are collected')
    p.add_argument('--max-files', type=int, default=None, help='Optional cap on number of files scanned (for debugging)')
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    ns = parse_args(argv or sys.argv[1:])

    # Ensure we're running from project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    os.chdir(project_root)

    files = list(iter_java_files(ns.input_dir))
    total_files = len(files) if ns.max_files is None else min(len(files), ns.max_files)

    cfg = ProcessConfig(
        input_dir=ns.input_dir,
        save_path=ns.save_path,
        out_format=ns.format,
        min_rows=ns.min_rows,
        max_files=ns.max_files,
    )

    rows = process_dataset(cfg)

    # Save
    save_output(rows, ns.save_path, ns.format)

    # Report
    print_summary(rows, total_files)

    # Optionally print a few samples
    if rows:
        print("")
        print("=== Sample row ===")
        r = rows[0]
        print(f"Type: {r['snippet_type']}")
        print(f"Name: {r['name']}")
        print(f"Code length: {len(r['code_snippet'])} chars")
        print(f"Code preview: {r['code_snippet'][:200]}...")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
