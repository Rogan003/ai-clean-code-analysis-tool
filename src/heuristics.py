from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from javalang.tree import MethodDeclaration

try:
    import javalang  # type: ignore
except Exception:  # javalang will be added to requirements
    javalang = None  # noqa: N816


_CAMEL_RE = re.compile(r"^[a-z]+(?:[A-Z][a-z0-9]*)*$")


def get_all_vars_from_method(method_obj: MethodDeclaration) -> List[str]:
    vars = []

    # Add parameter names
    if method_obj.parameters:
        for param in method_obj.parameters:
            vars.append(param.name)

    # Add local variable declarations
    for _, node in method_obj.filter(javalang.tree.LocalVariableDeclaration):
        for declarator in node.declarators:
            vars.append(declarator.name)

    return vars


def is_camel_case(name: str) -> bool:
    return bool(_CAMEL_RE.match(name or ""))


def count_special_chars(name: str) -> int:
    return sum(1 for c in name if not (c.isalnum()))


def comment_chars(src: str) -> int:
    total = 0
    for m in re.finditer(r"//.*", src):
        total += len(m.group(0))
    for m in re.finditer(r"/\*.*?\*/", src, flags=re.S):
        total += len(m.group(0))
    return total


def deepest_brace_nesting(method_obj: MethodDeclaration) -> int:
    # Statements that increase nesting depth
    nesting_nodes = (
        javalang.tree.IfStatement,
        javalang.tree.ForStatement,
        javalang.tree.WhileStatement,
        javalang.tree.DoStatement,
        javalang.tree.SwitchStatement,
        javalang.tree.TryStatement,
        javalang.tree.SynchronizedStatement,
        javalang.tree.BlockStatement,
    )
    
    max_depth = 0
    for path, node in method_obj.filter(nesting_nodes):
        # Count how many nesting nodes are in the path (ancestors)
        depth = sum(1 for ancestor in path if isinstance(ancestor, nesting_nodes))
        max_depth = max(max_depth, depth)
    
    return max_depth


def longest_line_len(src: str) -> int:
    return max((len(line) for line in src.splitlines() or [""]), default=0)


def cyclomatic_approx(method_obj: MethodDeclaration) -> int: # Unfortunately this won't work for recursive calls
    loop_nodes = (
        javalang.tree.ForStatement,
        javalang.tree.WhileStatement,
        javalang.tree.DoStatement,
    )
    
    max_depth = 0
    for path, node in method_obj.filter(loop_nodes):
        # Count how many loop nodes are in the path (ancestors)
        depth = sum(1 for ancestor in path if isinstance(ancestor, loop_nodes))
        max_depth = max(max_depth, depth)
    
    return max_depth


def returns_count(src: str) -> int:
    return len(re.findall(r"\breturn\b", src))


@dataclass
class HeuristicResult:
    features: List[float]
    feature_names: List[str]
    score: float  # 0..2 where 0 good, 1 warn, 2 bad
    label: int  # 0 green, 1 yellow, 2 red
    proba: List[float]  # pseudo-probabilities for 3 classes


# --- Feature bin helpers (0 good, 1 medium, 2 bad) ---

def bin_by_thresholds(val: float, t1: float, t2: float, min: float = 0.0) -> int:
    if val < min or val >= t2:
        return 2
    elif val < t1:
        return 0

    return 1


def map_bins_to_label(avg_bin: float) -> Tuple[int, float, List[float]]:
    # average of bins in [0,2]
    if avg_bin < 0.5:
        label = 0
    elif avg_bin < 1.3:
        label = 1
    else:
        label = 2
    # pseudo probabilities biased by avg_bin
    p_red = max(0.0, min(1.0, avg_bin / 2.0))
    p_green = max(0.0, min(1.0, (2.0 - avg_bin) / 2.0))
    p_yellow = max(0.0, 1.0 - p_red - p_green)
    s = p_green + p_yellow + p_red
    return label, avg_bin, [p_green / s, p_yellow / s, p_red / s]


# --- Public API ---

def method_heuristics(method_src: str, method_obj: MethodDeclaration) -> HeuristicResult:
    name_len = len(method_obj.name)
    name_special = count_special_chars(method_obj.name)
    name_camel = 0 if is_camel_case(method_obj.name) else 2

    all_vars = get_all_vars_from_method(method_obj) # TODO: Mean may not be the best possible way to calculate this
    variable_len = sum(len(var) for var in all_vars) / len(all_vars) if all_vars else 0
    variable_special = sum(count_special_chars(var) for var in all_vars) / len(all_vars) if all_vars else 0
    variable_camel = max(2, sum(1 for var in all_vars if not is_camel_case(var))) if all_vars else 0

    method_length = len([l for l in method_src.splitlines()])
    n_params = len(method_obj.parameters)
    comm = comment_chars(method_src)
    indent = deepest_brace_nesting(method_obj)
    long_line = longest_line_len(method_src)
    cyc = cyclomatic_approx(method_obj)
    rets = returns_count(method_src)

    bins = [
        bin_by_thresholds(name_len, 11, 21, 3),
        bin_by_thresholds(name_special, 1, 2),
        name_camel,
        bin_by_thresholds(variable_len, 8, 15),
        bin_by_thresholds(variable_special, 1, 2),
        variable_camel,
        bin_by_thresholds(method_length, 10, 31),
        bin_by_thresholds(n_params, 2, 5),
        bin_by_thresholds(comm, 50, 101),
        bin_by_thresholds(indent, 3, 5),
        bin_by_thresholds(long_line, 80, 111),
        bin_by_thresholds(cyc, 2, 3),
        bin_by_thresholds(rets, 2, 4),
    ]

    avg_bin = sum(bins) / len(bins)
    label, score, proba = map_bins_to_label(avg_bin)

    feats = [
        name_len,
        name_special,
        1 if name_camel == 0 else 0,
        variable_len,
        variable_special,
        1 if variable_camel == 0 else 0,
        method_length,
        n_params,
        comm,
        indent,
        long_line,
        cyc,
        rets,
    ]
    names = [
        "name_len",
        "name_special",
        "is_camel",
        "variable_len",
        "variable_special",
        "variable_camel",
        "loc",
        "n_params",
        "comment_chars",
        "nesting",
        "longest_line",
        "cyclomatic",
        "returns",
    ]
    return HeuristicResult(features=feats, feature_names=names, score=score, label=label, proba=proba)


def class_heuristics(class_src: str, class_name: str) -> HeuristicResult:
    # exclude getters/setters heuristically
    public_non_gs = len([m for m in re.findall(r"public\s+\w+\s+(\w+)\s*\(", class_src) if not (m.startswith("get") or m.startswith("set"))])
    fields = re.findall(r"(?:public|private|protected)\s+\w[\w<>,\s\[\]]*\s+(\w+)\s*(?:=|;)", class_src)
    n_vars = len(fields)
    prop_name_lens = [len(n) for n in fields]
    prop_name_special = [count_special_chars(n) for n in fields]

    avg_method_score_ref = None  # placeholder: external pipeline should inject this

    comm = comment_chars(class_src)
    name_len = len(class_name or "")
    name_special = count_special_chars(class_name)
    name_camel = 0 if (class_name and class_name[0].isupper() and is_camel_case(class_name[0].lower() + class_name[1:])) else 2

    # cohesion proxy: how many methods reference how many fields (very rough)
    methods_bodies = re.findall(r"\)\s*\{(.*?)\}", class_src, flags=re.S) # TODO: improve
    use_counts = 0
    for body in methods_bodies:
        used = 0
        for f in set(fields):
            if re.search(r"\b" + re.escape(f) + r"\b", body):
                used += 1
        use_counts += used
    cohesion = (use_counts / (len(methods_bodies) * max(1, len(set(fields))))) if methods_bodies else 1.0

    # TODO: Check field names, check if all heuristics are respected here

    bins = [
        bin_by_thresholds(name_len, 3, 21),
        bin_by_thresholds(int(sum(prop_name_lens) / max(1, len(prop_name_lens))), 3, 19),
        bin_by_thresholds(int(sum(prop_name_special) / max(1, len(prop_name_special))), 1, 2),
        bin_by_thresholds(public_non_gs, 5, 8),
        bin_by_thresholds(n_vars, 7, 11),
        # average method score to be injected; keep neutral (1) if unknown
        1,
        bin_by_thresholds(comm, 150, 301),
        bin_by_thresholds(1.0 - cohesion, 0.3, 0.6),  # lower cohesion -> worse
        name_camel,
    ]

    avg_bin = sum(bins) / len(bins)
    label, score, proba = map_bins_to_label(avg_bin)

    feats = [
        name_len,
        (sum(prop_name_lens) / max(1, len(prop_name_lens))) if prop_name_lens else 0,
        (sum(prop_name_special) / max(1, len(prop_name_special))) if prop_name_special else 0,
        public_non_gs,
        n_vars,
        comm,
        1.0 - cohesion,
        1 if name_camel == 0 else 0,
    ]
    names = [
        "name_len",
        "avg_field_name_len",
        "avg_field_name_special",
        "public_methods_no_gs",
        "n_fields",
        "comment_chars",
        "cohesion_inverted",
        "is_camel",
    ]
    return HeuristicResult(features=feats, feature_names=names, score=score, label=label, proba=proba)
