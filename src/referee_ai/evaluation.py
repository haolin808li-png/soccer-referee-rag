from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

DEFAULT_ORDERS = {
    "offence": ["No offence", "Between", "Offence"],
    "severity": ["1.0", "2.0", "3.0", "4.0"],
    "card": ["no_card", "yellow_card", "red_card"],
}


def _macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    scores = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _balanced_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    recalls = []
    for label in sorted(set(y_true)):
        indices = [i for i, value in enumerate(y_true) if value == label]
        recalls.append(sum(y_pred[i] == label for i in indices) / len(indices))
    return sum(recalls) / len(recalls) if recalls else 0.0


def ordinal_distance(y_true: Sequence[str], y_pred: Sequence[str], order: Sequence[str]) -> float:
    rank = {label: index for index, label in enumerate(order)}
    distances = [abs(rank[t] - rank[p]) for t, p in zip(y_true, y_pred) if t in rank and p in rank]
    return sum(distances) / len(distances) if distances else math.nan


def expected_calibration_error(
    correct: Sequence[bool], confidences: Sequence[float], bins: int = 10
) -> float:
    if len(correct) != len(confidences):
        raise ValueError("correct and confidences must have equal length")
    if not correct:
        return 0.0
    total = len(correct)
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = [i for i, value in enumerate(confidences) if low <= value <= high and (index == bins - 1 or value < high)]
        if selected:
            accuracy = sum(correct[i] for i in selected) / len(selected)
            confidence = sum(confidences[i] for i in selected) / len(selected)
            ece += len(selected) / total * abs(accuracy - confidence)
    return ece


def task_metrics(
    records: Sequence[Mapping[str, Any]], task: str, *, bins: int = 10,
    order: Sequence[str] | None = None,
) -> dict[str, float | int]:
    pairs = [(str(r["gold"][task]), str(r["prediction"][task]), r) for r in records
             if task in r.get("gold", {}) and task in r.get("prediction", {})
             and str(r["gold"][task]).lower() != "unknown"]
    y_true, y_pred = [p[0] for p in pairs], [p[1] for p in pairs]
    confidences = []
    for _, _, record in pairs:
        confidence = record.get("confidence", 1.0)
        if isinstance(confidence, Mapping):
            confidence = confidence.get(task, 1.0)
        confidences.append(min(1.0, max(0.0, float(confidence))))
    result: dict[str, float | int] = {
        "n": len(pairs),
        "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / len(pairs) if pairs else 0.0,
        "macro_f1": _macro_f1(y_true, y_pred),
        "balanced_accuracy": _balanced_accuracy(y_true, y_pred),
        "ece": expected_calibration_error([a == b for a, b in zip(y_true, y_pred)], confidences, bins),
    }
    ordinal_order = order or DEFAULT_ORDERS.get(task)
    if ordinal_order:
        result["ordinal_distance"] = ordinal_distance(y_true, y_pred, ordinal_order)
    return result


def bootstrap_confidence_intervals(
    records: Sequence[Mapping[str, Any]], metric: Callable[[Sequence[Mapping[str, Any]]], float],
    *, samples: int = 1000, confidence: float = 0.95, seed: int = 2026,
) -> dict[str, float]:
    if not records:
        return {"low": math.nan, "high": math.nan}
    rng = random.Random(seed)
    estimates = sorted(metric([records[rng.randrange(len(records))] for _ in records]) for _ in range(samples))
    alpha = (1 - confidence) / 2
    low = estimates[max(0, int(alpha * samples))]
    high = estimates[min(samples - 1, math.ceil((1 - alpha) * samples) - 1)]
    return {"low": low, "high": high}


def evaluate_predictions(
    records: Sequence[Mapping[str, Any]], *, bootstrap_samples: int = 0, seed: int = 2026
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"n_events": len(records), "tasks": {}}
    for task in ("offence", "severity", "card"):
        task_result = task_metrics(records, task)
        if bootstrap_samples:
            for name in ("macro_f1", "balanced_accuracy", "ordinal_distance"):
                if name in task_result:
                    task_result[f"{name}_ci"] = bootstrap_confidence_intervals(
                        records,
                        lambda sample, t=task, n=name: float(task_metrics(sample, t)[n]),
                        samples=bootstrap_samples,
                        seed=seed,
                    )
        metrics["tasks"][task] = task_result
    metrics["label_distribution"] = {
        task: dict(Counter(str(r.get("gold", {}).get(task, "missing")) for r in records))
        for task in ("offence", "severity", "card")
    }
    return metrics
