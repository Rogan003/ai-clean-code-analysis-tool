from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from javalang.tree import MethodDeclaration, ClassDeclaration

try:
    import javalang
except Exception:
    javalang = None

"""
ALL METHODS FOR CALCULATING HEURISTICS
HERE, WE CALCULATE HEURISTICS FOR A CLASS/METHOD AND FOR EVERY HEURISTIC WE DETERMINE THE CLASS OF THAT HEURISTIC
"""

_CAMEL_RE = re.compile(r"^[a-z]+(?:[A-Z][a-z0-9]*)*$")


def get_all_vars_from_method(method_obj: MethodDeclaration) -> List[str]:
    vars = []

    if method_obj.parameters:
        for param in method_obj.parameters:
            vars.append(param.name)

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
    label: int  # 0 green, 1 yellow, 2 red


# --- Feature bin helpers (0 good, 1 medium, 2 bad) ---

def bin_by_thresholds(val: float, t1: float, t2: float, min: float = 0.0) -> int:
    if val < min or val >= t2:
        return 2
    elif val < t1:
        return 0

    return 1


def map_bins_to_label(avg_bin: float) -> int:
    if avg_bin < 0.5:
        label = 0
    elif avg_bin < 1.3:
        label = 1
    else:
        label = 2

    return label


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
    label = map_bins_to_label(avg_bin)

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
    return HeuristicResult(features=feats, label=label)


def _is_getter_or_setter(method: MethodDeclaration) -> bool:
    name = method.name
    return (name.startswith("get") or name.startswith("set") or
            name.startswith("is") or name.startswith("has"))


def _get_public_methods_without_getters_setters(class_obj: ClassDeclaration) -> int:
    count = 0
    for member in class_obj.body:
        if isinstance(member, MethodDeclaration):
            is_public = not member.modifiers or 'public' in member.modifiers
            if is_public and not _is_getter_or_setter(member):
                count += 1
    return count


def _calculate_cohesion(class_obj: ClassDeclaration) -> float:
    """
    Calculate class cohesion using LCOM (Lack of Cohesion of Methods) approach.
    Returns a value between 0 (no cohesion) and 1 (perfect cohesion).
    """
    field_names = []
    for member in class_obj.body:
        if isinstance(member, javalang.tree.FieldDeclaration):
            for declarator in member.declarators:
                field_names.append(declarator.name)

    if not field_names:
        return 1.0

    methods = [m for m in class_obj.body if isinstance(m, MethodDeclaration)]
    if not methods:
        return 1.0

    # Count how many fields each method accesses
    total_field_accesses = 0
    for method in methods:
        fields_accessed = set()
        # Check all variable accesses in the method
        for _, node in method.filter(javalang.tree.MemberReference):
            if node.member in field_names:
                fields_accessed.add(node.member)
        total_field_accesses += len(fields_accessed)

    max_possible_accesses = len(methods) * len(field_names)
    cohesion = total_field_accesses / max_possible_accesses if max_possible_accesses > 0 else 1.0

    return cohesion


def class_heuristics(class_src: str, class_obj: ClassDeclaration, avg_method_score: float) -> HeuristicResult:
    if avg_method_score == "good":
        avg_method_score = 0
    elif avg_method_score == "changes_recommended":
        avg_method_score = 1
    else:
        avg_method_score = 2

    public_non_gs = _get_public_methods_without_getters_setters(class_obj)

    fields = []
    for member in class_obj.body:
        if isinstance(member, javalang.tree.FieldDeclaration):
            for declarator in member.declarators:
                fields.append(declarator.name)

    n_vars = len(fields)
    prop_name_lens = [len(n) for n in fields]
    prop_name_special = [count_special_chars(n) for n in fields]

    comm = comment_chars(class_src)
    name_len = len(class_obj.name or "")
    name_special = count_special_chars(class_obj.name)
    name_camel = 0 if (class_obj.name and class_obj.name[0].isupper() and is_camel_case(class_obj.name[0].lower() + class_obj.name[1:])) else 2

    cohesion = _calculate_cohesion(class_obj)

    bins = [
        bin_by_thresholds(name_len, 11, 21, 3),
        bin_by_thresholds(name_special, 1, 2),
        bin_by_thresholds(int(sum(prop_name_lens) / max(1, len(prop_name_lens))), 3, 19),
        bin_by_thresholds(int(sum(prop_name_special) / max(1, len(prop_name_special))), 1, 2),
        bin_by_thresholds(public_non_gs, 5, 8),
        bin_by_thresholds(n_vars, 7, 11),
        avg_method_score,
        bin_by_thresholds(comm, 150, 301),
        bin_by_thresholds(1.0 - cohesion, 0.3, 0.6),
        name_camel,
    ]

    avg_bin = sum(bins) / len(bins)
    label = map_bins_to_label(avg_bin)

    feats = [
        name_len,
        (sum(prop_name_lens) / max(1, len(prop_name_lens))) if prop_name_lens else 0,
        (sum(prop_name_special) / max(1, len(prop_name_special))) if prop_name_special else 0,
        public_non_gs,
        n_vars,
        comm,
        avg_method_score,
        1.0 - cohesion,
        1 if name_camel == 0 else 0,
    ]
    return HeuristicResult(features=feats, label=label)
