"""Compare time periods using shared stations and repeated Infomap fits."""

from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix

from citibike import graph


def common_graphs(slice_names, ids):
    """Find a shared station set connected in every period.

    Removing stations can disconnect another period, so repeat until stable.
    Coverage compares each final graph with its initial graph.
    """
    names = ["all", *slice_names]
    initial = {name: graph.load_graph(name, ids=ids) for name in names}
    common = np.asarray(ids)
    current = initial
    while True:
        shared = common
        for network in current.values():
            shared = np.intersect1d(shared, network.ids)
        if len(shared) < 2:
            raise ValueError("Fewer than two shared connected stations remain")
        if all(np.array_equal(network.ids, shared) for network in current.values()):
            break
        common = shared
        current = {name: graph.load_graph(name, ids=common) for name in names}

    coverage = pd.DataFrame([
        {
            "slice": name,
            "initial_stations": len(initial[name].ids),
            "shared_stations": len(current[name].ids),
            "initial_trips": int(initial[name].W.sum()),
            "shared_trips": int(current[name].W.sum()),
            "trip_coverage": float(current[name].W.sum() / initial[name].W.sum()),
        }
        for name in names
    ]).set_index("slice")
    return current, coverage


def fit_repeated(networks, seeds, trials):
    """Use the same seeds and Infomap settings for every period."""
    runs = {}
    for name, network in networks.items():
        runs[name] = {}
        for seed in seeds:
            print(f"Infomap {name}, seed {seed}, trials {trials}", flush=True)
            runs[name][seed] = graph.infomap(network.W, seed=seed, trials=trials)
    return runs


def align_partition(reference, labels):
    """Match labels one-to-one to maximize shared stations.

    Extra groups get new IDs. ARI can compare the original labels directly.
    """
    reference_values = np.unique(reference)
    label_values = np.unique(labels)
    overlap = contingency_matrix(reference, labels)
    rows, columns = linear_sum_assignment(-overlap)
    mapping = {label_values[col]: reference_values[row] for row, col in zip(rows, columns)}
    next_label = int(reference_values.max()) + 1
    for label in label_values:
        if label not in mapping:
            mapping[label] = next_label
            next_label += 1
    return np.array([mapping[label] for label in labels])


def representative(partitions):
    """Choose the run with the highest mean agreement with the other runs."""
    seeds = list(partitions)
    agreement = np.array([
        [adjusted_rand_score(partitions[a], partitions[b]) for b in seeds]
        for a in seeds
    ])
    return partitions[seeds[int(agreement.mean(axis=1).argmax())]]


def compare_runs(networks, runs):
    """Measure agreement within and across periods, plus retention for each seed."""
    within = []
    across = []
    retention = []
    annual_runs = runs["all"]
    for name, partitions in runs.items():
        for a, b in combinations(partitions, 2):
            within.append({
                "slice": name, "seed_a": a, "seed_b": b,
                "ari": adjusted_rand_score(partitions[a], partitions[b]),
            })
        if name == "all":
            continue
        for annual_seed, slice_seed in product(annual_runs, partitions):
            across.append({
                "slice": name, "annual_seed": annual_seed, "slice_seed": slice_seed,
                "ari": adjusted_rand_score(annual_runs[annual_seed], partitions[slice_seed]),
            })
        for seed, labels in partitions.items():
            fixed = graph.flow_retention(networks[name].W, annual_runs[seed])
            fitted = graph.flow_retention(networks[name].W, labels)
            retention.append({
                "slice": name, "seed": seed,
                "annual_k": len(np.unique(annual_runs[seed])),
                "slice_k": len(np.unique(labels)),
                "annual_retention": fixed, "slice_retention": fitted,
                "gain_pp": 100 * (fitted - fixed),
            })
    return pd.DataFrame(within), pd.DataFrame(across), pd.DataFrame(retention)
