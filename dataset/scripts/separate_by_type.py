#!/usr/bin/env python3
"""
Separate d4j_extracted.csv into separate files for classes and methods.

- Methods file: Contains only code_snippet and score
- Classes file: Contains code_snippet, score, and average_method_score

Usage:
  python dataset/scripts/separate_by_type.py \
      --input dataset/data/d4j_extracted.csv \
      --methods-output dataset/data/official_dataset_methods.csv \
      --classes-output dataset/data/official_dataset_classes.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


def score_to_points(score: str) -> Optional[int]:
    """Convert score string to numeric points.

    good = 0, changes_recommended = 1, changes_required = 2
    Returns None if score is invalid.
    """
    score = score.strip().lower()
    if score == "good":
        return 0
    elif score == "changes_recommended":
        return 1
    elif score == "changes_required":
        return 2
    return None


def points_to_score(avg_points: float) -> str:
    """Convert average points to score label.

    < 0.5 => good
    < 1.3 => changes_recommended
    >= 1.3 => changes_required
    """
    if avg_points < 0.5:
        return "good"
    elif avg_points < 1.3:
        return "changes_recommended"
    else:
        return "changes_required"


def calculate_average_method_score(method_scores: List[str]) -> Optional[str]:
    """Calculate average score from list of method scores.

    Returns the calculated average score label, or None if no valid scores.
    """
    if not method_scores:
        return None

    points = [score_to_points(s) for s in method_scores]
    valid_points = [p for p in points if p is not None]

    if not valid_points:
        return None

    avg = sum(valid_points) / len(valid_points)
    return points_to_score(avg)


def process_csv(input_path: str, methods_output: str, classes_output: str) -> None:
    """Process the CSV file and separate into methods and classes files."""

    methods_rows = []
    classes_rows = []

    print(f"Reading {input_path}...")

    # First pass: read all rows and group methods by their preceding class
    all_rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading rows"):
            all_rows.append(row)

    print("Processing rows and calculating average method scores...")

    current_class_idx = None
    class_method_scores = {}  # Maps class index to list of method scores

    # Process rows and collect method scores for each class
    for idx, row in enumerate(tqdm(all_rows, desc="Processing rows")):
        row_type = row.get('type', '').strip().lower()
        code_snippet = row.get('code_snippet', '')
        score = row.get('score', '')

        if row_type == 'class':
            current_class_idx = idx
            class_method_scores[idx] = []
        elif row_type == 'method' and current_class_idx is not None:
            # Associate this method with the current class
            class_method_scores[current_class_idx].append(score)

    # Second pass: create output rows with calculated averages
    for idx, row in enumerate(all_rows):
        row_type = row.get('type', '').strip().lower()
        code_snippet = row.get('code_snippet', '')
        score = row.get('score', '')

        if row_type == 'method':
            methods_rows.append({
                'code_snippet': code_snippet,
                'score': score
            })
        elif row_type == 'class':
            method_scores = class_method_scores.get(idx, [])
            avg_method_score = calculate_average_method_score(method_scores)
            classes_rows.append({
                'code_snippet': code_snippet,
                'score': score,
                'average_method_score': avg_method_score if avg_method_score is not None else ''
            })

    # Write methods file
    print(f"\nWriting {len(methods_rows)} methods to {methods_output}...")
    os.makedirs(os.path.dirname(methods_output) or ".", exist_ok=True)
    with open(methods_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['code_snippet', 'score'])
        writer.writeheader()
        writer.writerows(methods_rows)

    # Write classes file
    print(f"Writing {len(classes_rows)} classes to {classes_output}...")
    os.makedirs(os.path.dirname(classes_output) or ".", exist_ok=True)
    with open(classes_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['code_snippet', 'score', 'average_method_score'])
        writer.writeheader()
        writer.writerows(classes_rows)

    print("\n=== Summary ===")
    print(f"Total methods: {len(methods_rows)}")
    print(f"Total classes: {len(classes_rows)}")
    print(f"Total rows processed: {len(methods_rows) + len(classes_rows)}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Separate CSV by type into methods and classes files.')
    p.add_argument('--input', default='../data/d4j_extracted.csv',
                   help='Input CSV file path')
    p.add_argument('--methods-output', default='../data/d4j_classes.csv',
                   help='Output file for methods')
    p.add_argument('--classes-output', default='../data/d4j_methods.csv',
                   help='Output file for classes')
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    ns = parse_args(argv or sys.argv[1:])

    # Ensure input file exists
    if not os.path.exists(ns.input):
        print(f"Error: Input file '{ns.input}' not found.", file=sys.stderr)
        return 1

    process_csv(ns.input, ns.methods_output, ns.classes_output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
