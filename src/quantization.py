"""
quantization.py
---------------
M4 / M5 — Dynamic INT8 quantization of student models.

Applies PyTorch's post-training dynamic quantization to convert linear layers
from FP32 to INT8, reducing model size and improving CPU inference throughput.

Phase 1: placeholder — full implementation added in Phase 5.
"""

# TODO (Phase 5): implement quantize_model(), save_quantized_model(), load_quantized_model()


def quantize_model(model, dtype=None):
    """Apply dynamic INT8 quantization to a PyTorch model.

    Targets torch.nn.Linear layers, which dominate Transformer inference cost.

    Args:
        model: A trained PyTorch model (FP32).
        dtype: Quantization dtype — defaults to torch.qint8.

    Returns:
        Quantized model (CPU only; dynamic quantization is CPU-bound).
    """
    raise NotImplementedError("Implemented in Phase 5 — src/quantization.py")


def save_quantized_model(quantized_model, save_dir: str):
    """Serialize a quantized model to disk.

    Args:
        quantized_model: Output of quantize_model().
        save_dir:        Target directory path.
    """
    raise NotImplementedError("Implemented in Phase 5 — src/quantization.py")


def load_quantized_model(save_dir: str):
    """Load a previously quantized model from disk.

    Args:
        save_dir: Directory that was used in save_quantized_model().

    Returns:
        Quantized PyTorch model ready for CPU inference.
    """
    raise NotImplementedError("Implemented in Phase 5 — src/quantization.py")
