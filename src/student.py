"""
student.py
----------
M2 — Supervised DistilBERT-base-uncased baseline (FP32).

Phase 1: placeholder — full implementation added in Phase 3.
"""

# TODO (Phase 3): implement StudentModel, train_student_baseline(), load_student()


class StudentModel:
    """Wrapper around DistilBertForSequenceClassification for SST-2.

    Attributes:
        model_name: Hugging Face Hub identifier (distilbert-base-uncased).
        num_labels: Number of classification labels (2 for SST-2).
    """

    def __init__(self, model_name: str, num_labels: int):
        raise NotImplementedError("Implemented in Phase 3 — src/student.py")


def train_student_baseline(student_model, train_loader, val_loader, config):
    """Fine-tune DistilBERT directly on SST-2 without a teacher (M2 baseline).

    Args:
        student_model: StudentModel instance.
        train_loader:  Training DataLoader.
        val_loader:    Validation DataLoader.
        config:        Project config module.
    """
    raise NotImplementedError("Implemented in Phase 3 — src/student.py")


def load_student(checkpoint_dir: str, config):
    """Load a saved student checkpoint.

    Args:
        checkpoint_dir: Path to the saved model directory.
        config:         Project config module.

    Returns:
        A loaded StudentModel ready for inference or distillation.
    """
    raise NotImplementedError("Implemented in Phase 3 — src/student.py")
