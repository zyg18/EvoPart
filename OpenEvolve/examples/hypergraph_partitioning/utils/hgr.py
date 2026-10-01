"""
.hgr I/O and objective computation, in Python.

This is the *scoring authority*. The evolved C++ program prints its own RESULT
line, but the evaluator never trusts it: every candidate is scored by re-reading
the partition vector it wrote and recomputing cut and balance here. A candidate
therefore cannot improve its score by changing how the C++ side measures itself.
"""

import math
import os
from functools import lru_cache

# Balance constraint: cap = floor((1 + eps) * ceil(n / k)) vertices per part
def capacity_for(n, k, eps):
    return math.floor((1.0 + eps) * math.ceil(n / k))


@lru_cache(maxsize=32)
def read_hgr(path):
    """Return (num_vertices, nets) with nets as a tuple of tuples of 1-indexed ids."""
    with open(path) as f:
        header = None
        nets = []
        for line in f:
            if not line.strip() or line.startswith("%"):
                continue
            if header is None:
                parts = line.split()
                header = (int(parts[0]), int(parts[1]))
                continue
            nets.append(tuple(int(x) for x in line.split()))
            if len(nets) == header[0]:
                break
    m, n = header
    if len(nets) != m:
        raise ValueError(f"{path}: expected {m} hyperedges, read {len(nets)}")
    return n, tuple(nets)


def read_partition(path, n):
    """Read one part id per line; raises if the file is malformed."""
    with open(path) as f:
        part = [int(x) for x in f.read().split()]
    if len(part) != n:
        raise ValueError(f"{path}: expected {n} part ids, got {len(part)}")
    return part


def evaluate_partition(hgr_path, part, k, eps):
    """
    Recompute the objective from scratch.

    `part` is 0-indexed by vertex (part[i] is the part of vertex i+1).
    Returns a dict; `feasible` is False if any vertex is misassigned or any part
    exceeds its capacity.
    """
    n, nets = read_hgr(hgr_path)
    if len(part) != n:
        return {"feasible": False, "reason": f"partition has {len(part)} entries, need {n}"}

    sizes = [0] * k
    for p in part:
        if not (0 <= p < k):
            return {"feasible": False, "reason": f"part id {p} outside [0,{k})"}
        sizes[p] += 1

    cut = 0
    soed = 0
    for e in nets:
        touched = {part[v - 1] for v in e}
        if len(touched) > 1:
            cut += 1
        soed += len(touched) - 1

    cap = capacity_for(n, k, eps)
    largest = max(sizes)
    return {
        "feasible": largest <= cap,
        "reason": "" if largest <= cap else f"part of {largest} exceeds capacity {cap}",
        "cut": cut,
        "soed": soed,
        "sizes": sizes,
        "capacity": cap,
        "imbalance": largest / (n / k) - 1.0,
    }


def geometric_mean(values):
    """Geometric mean, guarding against zeros so one failure cannot annihilate the rest."""
    if not values:
        return 0.0
    acc = sum(math.log(max(v, 1e-3)) for v in values)
    return math.exp(acc / len(values))


def dataset_path(name, dataset_dir):
    return os.path.join(dataset_dir, f"{name}.hgr")
