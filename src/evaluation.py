"""
evaluation.py
-------------
Unified evaluation utilities used across all five experimental models.

Computes: accuracy, F1, inference latency, and model size on disk.
Produces comparison tables and plots saved to results/.

Phase 1: placeholder — full implementation added in Phase 6.
"""

# TODO (Phase 6): implement evaluate_model(), benchmark_inference(), compare_models(), plot_results()


def evaluate_model(model, data_loader, device, model_name: str = "model"):
    """Compute accuracy and macro-F1 on a held-out split.

    Args:
        model:       A trained PyTorch model.
        data_loader: DataLoader for the evaluation split.
        device:      torch.device to run inference on.
        model_name:  Label used in returned metrics dict.

    Returns:
        dict with keys: model_name, accuracy, f1, num_samples.
    """
    raise NotImplementedError("Implemented in Phase 6 — src/evaluation.py")


def benchmark_inference(model, data_loader, device, n_runs: int = 3):
    """Measure mean inference latency per batch over n_runs passes.

    Args:
        model:       A trained (or quantized) PyTorch model.
        data_loader: DataLoader — typically the validation split.
        device:      torch.device for inference.
        n_runs:      Number of timing repetitions to average.

    Returns:
        dict with keys: mean_latency_ms, std_latency_ms, throughput_samples_per_sec.
    """
    raise NotImplementedError("Implemented in Phase 6 — src/evaluation.py")


def get_model_size_mb(model) -> float:
    """Return the in-memory parameter footprint of a model in megabytes.

    Args:
        model: Any PyTorch nn.Module.

    Returns:
        Size in MB (float).
    """
    raise NotImplementedError("Implemented in Phase 6 — src/evaluation.py")


def compare_models(results: list) -> None:
    """Print a formatted comparison table for all five models.

    Args:
        results: List of dicts returned by evaluate_model() /
                 benchmark_inference() — one per model.
    """
    raise NotImplementedError("Implemented in Phase 6 — src/evaluation.py")


def plot_results(results: list, save_dir: str) -> None:
    """Generate and save bar charts comparing accuracy, F1, latency, and size.

    Args:
        results:  List of metric dicts (one per model).
        save_dir: Directory to write PNG files into.
    """
    raise NotImplementedError("Implemented in Phase 6 — src/evaluation.py")
