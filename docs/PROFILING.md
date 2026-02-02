# Performance Profiling Checklist

When preparing for large-scale cluster runs, verify the following:

## 1. Data Throughput
- [ ] Monitor GPU utilization (`nvidia-smi -l 1`). If volatile < 80%, data loading might be the bottleneck.
- [ ] Check `num_workers`. Set `num_workers > 0` (typically 4-8 per GPU) in `config.data`.
- [ ] Verify `pin_memory=True` in `config.data`.
- [ ] Ensure `MetaIterableDataset` generation is not CPU-bound (e.g., heavy `networkx` ops in `sample_graph`).

## 2. Distributed Overheads
- [ ] Check `nccl` initialization time in logs.
- [ ] Verify `find_unused_parameters=False` in DDP wrapper (set in `pipe.py`) unless absolutely necessary.
- [ ] Ensure validation interval is not too frequent (e.g., `val_check_interval` >= 1000 steps).

## 3. Memory Usage
- [ ] Profile peak VRAM usage. Enable `tf32` (TensorFloat-32) on Ampere+ GPUs for 3x speedup on matmuls (`trainer.tf32=true`).
- [ ] Use Mixed Precision (`trainer.amp=true`) to halve memory bandwidth usage.

## 4. Artifact I/O
- [ ] For inference, ensure `output_dir` is on a fast filesystem (e.g., scratch/SSD), not NFS.
- [ ] Use `cache_compress=true` (gzip) for large graph dumps to save I/O bandwidth.
