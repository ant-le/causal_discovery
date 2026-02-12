from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
from causica.datasets.dataset import CausalDataset
from causica.datasets.variables import Variables
from causica.models.bayesdag.bayesdag_linear import BayesDAGLinear
from causica.models.bayesdag.bayesdag_nonlinear import BayesDAGNonLinear

log = logging.getLogger(__name__)


def _build_dataset(
    train_data: np.ndarray,
    train_mask: np.ndarray,
    variables: Variables,
    seed: int,
) -> CausalDataset:
    num_nodes = train_data.shape[1]
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    subgraph_mask = np.ones((num_nodes, num_nodes), dtype=np.float32)
    graph_args: Dict[str, Any] = {
        "num_variables": int(num_nodes),
        "exp_edges": float(adjacency.sum()),
        "exp_edges_per_node": float(adjacency.sum()) / max(1, num_nodes),
        "graph_type": "unknown",
        "seed": int(seed),
    }

    return CausalDataset(
        train_data=train_data,
        train_mask=train_mask,
        adjacency_data=adjacency,
        subgraph_data=subgraph_mask,
        intervention_data=None,
        counterfactual_data=None,
        val_data=None,
        val_mask=None,
        test_data=train_data,
        test_mask=train_mask,
        variables=variables,
        data_split=None,
        held_out_interventions=None,
        true_posterior=None,
        graph_args=graph_args,
    )


def _variables_dict_from_data(data: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    variables = []
    for idx in range(data.shape[1]):
        col = data[:, idx]
        col_mask = mask[:, idx].astype(bool)
        if col_mask.any():
            col_min = float(col[col_mask].min())
            col_max = float(col[col_mask].max())
        else:
            col_min = 0.0
            col_max = 0.0
        variables.append(
            {
                "name": f"Column {idx}",
                "type": "continuous",
                "lower": col_min,
                "upper": col_max,
                "query": True,
                "target": False,
                "always_observed": True,
            }
        )
    return {
        "variables": variables,
        "auxiliary_variables": [],
        "used_cols": list(range(data.shape[1])),
    }


def _validate_data(data: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(data).all():
        raise ValueError("BayesDAG input data contains NaN or inf values.")
    if not np.isfinite(mask).all():
        raise ValueError("BayesDAG input mask contains NaN or inf values.")
    if data.ndim != 2:
        raise ValueError("BayesDAG input data must have shape (samples, variables).")
    if mask.shape != data.shape:
        raise ValueError("BayesDAG input mask must match data shape.")
    return data, mask


def _log_data_stats(data: np.ndarray, mask: np.ndarray) -> None:
    observed = mask.astype(bool)
    if observed.any():
        data_obs = data[observed]
        log.info(
            "BayesDAG data stats: n=%d, d=%d, min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
            data.shape[0],
            data.shape[1],
            float(data_obs.min()),
            float(data_obs.max()),
            float(data_obs.mean()),
            float(data_obs.std()),
        )
    else:
        log.warning("BayesDAG data mask has no observed entries.")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _build_model(config: Dict[str, Any], variables: Variables) -> Any:
    variant = config["variant"]
    model_cls = BayesDAGLinear if variant == "linear" else BayesDAGNonLinear

    kwargs = {
        "model_id": f"bayesdag_{variant}",
        "variables": variables,
        "save_dir": config["save_dir"],
        "device": _resolve_device(config.get("device", "auto")),
        "lambda_sparse": config["lambda_sparse"],
        "num_chains": config["num_chains"],
        "sinkhorn_n_iter": config["sinkhorn_n_iter"],
        "scale_noise": config["scale_noise"],
        "scale_noise_p": config["scale_noise_p"],
        "VI_norm": config["vi_norm"],
        "input_perm": config["input_perm"],
        "sparse_init": config["sparse_init"],
    }
    if variant == "nonlinear":
        kwargs["norm_layers"] = config["norm_layers"]
        kwargs["res_connection"] = config["res_connection"]

    return model_cls(**kwargs)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    input_data = np.load(args.input)
    data = input_data["data"].astype(np.float32, copy=False)
    mask = input_data["mask"].astype(np.float32, copy=False)
    _validate_data(data, mask)
    _log_data_stats(data, mask)

    variables = Variables.create_from_data_and_dict(
        data, mask, _variables_dict_from_data(data, mask)
    )
    dataset = _build_dataset(data, mask, variables, seed=int(config["seed"]))

    model = _build_model(config["model"], variables)
    if bool(config.get("train", {}).get("skip_evaluation", False)):
        model.evaluate_metrics = lambda *args, **kwargs: None

    start_train = time.perf_counter()
    model.run_train(dataset, train_config_dict=config["train"])
    train_s = time.perf_counter() - start_train

    start_sample = time.perf_counter()
    graph_samples, _ = model.get_adj_matrix_tensor(samples=int(config["num_samples"]))
    sample_s = time.perf_counter() - start_sample

    log.info(
        "BayesDAG timing: train=%.2fs, sample=%.2fs, total=%.2fs",
        train_s,
        sample_s,
        train_s + sample_s,
    )
    np.savez(config["output"], graph_samples=graph_samples.detach().cpu().numpy())


if __name__ == "__main__":
    main()
