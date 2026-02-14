from __future__ import annotations

from unittest.mock import patch

import torch

from causal_meta.runners.metrics.graph import Metrics


def test_metrics_handler_compute() -> None:
    metrics = Metrics(metrics=["e-shd", "e-edgef1"])
    target = torch.tensor([[[0, 1], [0, 0]]]).float()
    samples = torch.tensor([[[[0, 1], [0, 0]]], [[[0, 0], [0, 0]]]]).float()

    results = metrics.compute(None, target, samples=samples)
    assert "e-shd" in results
    assert "e-edgef1" in results
    assert results["e-shd"] == 0.5
    assert results["e-edgef1"] == 0.5


@patch("torch.distributed.is_available")
@patch("torch.distributed.is_initialized")
@patch("torch.distributed.all_reduce")
def test_metrics_sync_distributed(mock_all_reduce, mock_init, mock_avail) -> None:
    mock_avail.return_value = True
    mock_init.return_value = True

    metrics_handler = Metrics()
    local_metrics = {"e-shd": 1.0, "graph_nll": 5.0}

    synced = metrics_handler.sync(local_metrics)

    assert mock_all_reduce.called
    assert "e-shd" in synced
    assert "graph_nll" in synced


@patch("torch.distributed.is_available")
@patch("torch.distributed.is_initialized")
@patch("torch.distributed.get_world_size")
@patch("torch.distributed.all_gather_object")
def test_metrics_gather_distributed(
    mock_all_gather, mock_world_size, mock_init, mock_avail
) -> None:
    mock_avail.return_value = True
    mock_init.return_value = True
    mock_world_size.return_value = 2

    rank0_data = {"e-shd": [0.5, 1.0]}
    rank1_data = {"e-shd": [1.5, 2.0]}

    def side_effect(gathered_list, local_obj):
        _ = local_obj
        gathered_list[0] = rank0_data
        gathered_list[1] = rank1_data
        return None

    mock_all_gather.side_effect = side_effect

    metrics_handler = Metrics()
    gathered = metrics_handler.gather(rank0_data)

    assert gathered["e-shd"] == [0.5, 1.0, 1.5, 2.0]
