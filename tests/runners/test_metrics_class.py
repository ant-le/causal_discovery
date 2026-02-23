import torch
from causal_meta.runners.metrics.graph import Metrics


def test_metrics_class_stateful():
    metrics = Metrics(metrics=["e-shd", "e-edgef1"])

    # Batch 1
    target1 = torch.tensor([[[0, 1], [0, 0]]])  # (1, 2, 2)
    pred1 = torch.tensor([[[[0, 1], [0, 0]]]])  # (1, 1, 2, 2) Perfect match

    metrics.update(target1, pred1, prefix="ds1")

    # Batch 2
    target2 = torch.tensor([[[0, 1], [0, 0]]])
    pred2 = torch.tensor([[[[0, 0], [1, 0]]]])  # (1, 1, 2, 2) Reversed -> SHD=2, F1=0

    metrics.update(target2, pred2, prefix="ds1")

    # Check raw results
    raw = metrics.get_raw_results()
    assert "ds1/e-shd" in raw
    assert len(raw["ds1/e-shd"]) == 2
    assert raw["ds1/e-shd"] == [0.0, 2.0]
    assert raw["e-shd"] == [0.0, 2.0]

    # Check compute (summary)
    summary = metrics.compute(summary_stats=True)
    assert summary["e-shd_mean"] == 1.0
    assert summary["ds1/e-shd_mean"] == 1.0

    # Check reset
    metrics.reset()
    assert len(metrics.get_raw_results()) == 0


def test_metrics_compute_simple():
    metrics = Metrics(metrics=["e-shd"])
    target = torch.tensor([[[0, 1], [0, 0]]])
    pred = torch.tensor([[[[0, 1], [0, 0]]]])

    metrics.update(target, pred)

    res = metrics.compute(summary_stats=False)
    assert res["e-shd"] == 0.0
