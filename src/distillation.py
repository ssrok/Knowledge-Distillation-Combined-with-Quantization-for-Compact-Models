"""
distillation.py
---------------
M3 — Knowledge-distilled DistilBERT student (FP32).

Implements the combined KL-divergence + cross-entropy distillation loss and
the training loop that uses soft teacher labels alongside hard ground-truth labels.

Phase 1: placeholder — full implementation added in Phase 4.
"""

# TODO (Phase 4): implement DistillationLoss, train_distilled_student()


class DistillationLoss:
    """Combined distillation loss.

    loss = alpha * T^2 * KLDiv(student_soft, teacher_soft)
           + (1 - alpha) * CrossEntropy(student_logits, hard_labels)

    Args:
        temperature: Softening factor T applied to both teacher and student logits.
        alpha:       Weight of the KL-divergence term (0 → pure CE, 1 → pure KL).
    """

    def __init__(self, temperature: float, alpha: float):
        raise NotImplementedError("Implemented in Phase 4 — src/distillation.py")

    def __call__(self, student_logits, teacher_logits, labels):
        raise NotImplementedError("Implemented in Phase 4 — src/distillation.py")


def train_distilled_student(
    student_model,
    teacher_model,
    train_loader,
    val_loader,
    config,
):
    """Train the student using knowledge distillation from the teacher (M3).

    Args:
        student_model: Initialised StudentModel (untrained or pre-fine-tuned).
        teacher_model: Frozen TeacherModel checkpoint (M1).
        train_loader:  Training DataLoader.
        val_loader:    Validation DataLoader.
        config:        Project config module.
    """
    raise NotImplementedError("Implemented in Phase 4 — src/distillation.py")
