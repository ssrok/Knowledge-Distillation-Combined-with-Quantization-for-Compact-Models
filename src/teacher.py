"""
teacher.py
----------
M1 — Fine-tuned BERT-base-uncased teacher model (FP32).

Phase 1: placeholder — full implementation added in Phase 2.
"""

# TODO (Phase 2): implement TeacherModel, train_teacher(), load_teacher()


class TeacherModel:
    """Wrapper around BertForSequenceClassification for SST-2.

    Attributes:
        model_name: Hugging Face Hub identifier (bert-base-uncased).
        num_labels: Number of classification labels (2 for SST-2).
    """

    def __init__(self, model_name: str, num_labels: int):
        raise NotImplementedError("Implemented in Phase 2 — src/teacher.py")


def train_teacher(teacher_model, train_loader, val_loader, config):
    """Fine-tune the teacher on SST-2 and save the best checkpoint.

    Args:
        teacher_model: TeacherModel instance.
        train_loader:  Training DataLoader.
        val_loader:    Validation DataLoader.
        config:        Project config module.
    """
    raise NotImplementedError("Implemented in Phase 2 — src/teacher.py")


def load_teacher(checkpoint_dir: str, config):
    """Load a saved teacher checkpoint for inference or distillation.

    Args:
        checkpoint_dir: Path to the saved model directory.
        config:         Project config module.

    Returns:
        A loaded TeacherModel ready for inference.
    """
    raise NotImplementedError("Implemented in Phase 2 — src/teacher.py")
