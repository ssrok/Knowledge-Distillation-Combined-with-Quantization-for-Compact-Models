"""
setup_verify.py
---------------
Lightweight environment validation script for Phase 1.

Checks:
  1. Python version
  2. Required package imports (without downloading models)
  3. Folder structure completeness
  4. Configuration values
  5. Device availability (CPU / CUDA)

Run with:
    python setup_verify.py
"""

import sys
import importlib
from pathlib import Path

PASS  = "  [PASS]"
FAIL  = "  [FAIL]"
WARN  = "  [WARN]"
INFO  = "  [INFO]"
SEP   = "-" * 60


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def check_python_version() -> bool:
    section("1. Python Version")
    major, minor = sys.version_info[:2]
    print(f"  Python {major}.{minor}.{sys.version_info.micro}")
    if (major, minor) >= (3, 8):
        print(f"{PASS}  Python >= 3.8 required.")
        return True
    else:
        print(f"{FAIL}  Python 3.8+ required.")
        return False


def check_imports() -> bool:
    section("2. Package Imports")
    packages = {
        "torch":        "PyTorch",
        "transformers": "Hugging Face Transformers",
        "datasets":     "Hugging Face Datasets",
        "sklearn":      "scikit-learn",
        "numpy":        "NumPy",
        "pandas":       "Pandas",
        "matplotlib":   "Matplotlib",
        "streamlit":    "Streamlit",
        "tqdm":         "tqdm",
        "evaluate":     "HF evaluate",
    }
    all_ok = True
    for pkg, label in packages.items():
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "?")
            print(f"{PASS}  {label:<35} {version}")
        except ImportError:
            print(f"{FAIL}  {label:<35} NOT FOUND — run: pip install -r requirements.txt")
            all_ok = False
    return all_ok


def check_folder_structure() -> bool:
    section("3. Folder Structure")
    root = Path(__file__).resolve().parent
    required_dirs = [
        "data",
        "models/teacher",
        "models/student_baseline",
        "models/student_distilled",
        "models/student_quantized",
        "notebooks",
        "src",
        "results",
        "app",
    ]
    required_files = [
        "requirements.txt",
        ".gitignore",
        "README.md",
        "src/__init__.py",
        "src/config.py",
        "src/dataset.py",
        "src/teacher.py",
        "src/student.py",
        "src/distillation.py",
        "src/quantization.py",
        "src/evaluation.py",
        "src/utils.py",
        "app/streamlit_app.py",
        "notebooks/01_teacher_training.ipynb",
        "notebooks/02_student_baseline.ipynb",
        "notebooks/03_knowledge_distillation.ipynb",
        "notebooks/04_quantization.ipynb",
        "notebooks/05_evaluation.ipynb",
    ]
    all_ok = True
    for d in required_dirs:
        p = root / d
        if p.is_dir():
            print(f"{PASS}  dir   {d}")
        else:
            print(f"{FAIL}  dir   {d}  MISSING")
            all_ok = False
    for f in required_files:
        p = root / f
        if p.is_file():
            print(f"{PASS}  file  {f}")
        else:
            print(f"{FAIL}  file  {f}  MISSING")
            all_ok = False
    return all_ok


def check_config() -> bool:
    section("4. Configuration Values")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from src import config  # noqa: PLC0415

        checks = {
            "TEACHER_MODEL_NAME":  (config.TEACHER_MODEL_NAME,  "bert-base-uncased"),
            "STUDENT_MODEL_NAME":  (config.STUDENT_MODEL_NAME,  "distilbert-base-uncased"),
            "DATASET_NAME":        (config.DATASET_NAME,         "glue"),
            "DATASET_CONFIG":      (config.DATASET_CONFIG,       "sst2"),
            "NUM_LABELS":          (config.NUM_LABELS,            2),
            "MAX_SEQ_LENGTH":      (config.MAX_SEQ_LENGTH,        128),
            "BATCH_SIZE":          (config.BATCH_SIZE,            16),
            "NUM_EPOCHS":          (config.NUM_EPOCHS,            2),
            "LEARNING_RATE":       (config.LEARNING_RATE,         2e-5),
            "TEMPERATURE":         (config.TEMPERATURE,           2.0),
            "ALPHA":               (config.ALPHA,                 0.5),
            "RANDOM_SEED":         (config.RANDOM_SEED,           42),
        }
        all_ok = True
        for name, (actual, expected) in checks.items():
            if actual == expected:
                print(f"{PASS}  {name:<25} = {actual}")
            else:
                print(f"{FAIL}  {name:<25} = {actual}  (expected {expected})")
                all_ok = False
        return all_ok
    except Exception as exc:
        print(f"{FAIL}  Could not import src.config: {exc}")
        return False


def check_device() -> bool:
    section("5. Device Availability")
    try:
        import torch  # noqa: PLC0415
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{INFO}  PyTorch version : {torch.__version__}")
        print(f"{INFO}  CUDA available  : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"{INFO}  GPU             : {torch.cuda.get_device_name(0)}")
            print(f"{INFO}  CUDA version    : {torch.version.cuda}")
        print(f"{INFO}  Active device   : {device}")
        print(f"{PASS}  Device selection OK — will use: {device}")
        return True
    except ImportError:
        print(f"{WARN}  PyTorch not installed — device check skipped.")
        return False


def main() -> None:
    print("\n" + "=" * 60)
    print("  Knowledge Distillation + Quantization — Setup Verification")
    print("=" * 60)

    results = {
        "Python version":    check_python_version(),
        "Package imports":   check_imports(),
        "Folder structure":  check_folder_structure(),
        "Configuration":     check_config(),
        "Device":            check_device(),
    }

    section("Summary")
    all_pass = True
    for check, ok in results.items():
        status = PASS if ok else FAIL
        print(f"{status}  {check}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("  ✅  All checks passed — Phase 1 setup is complete.")
    else:
        print("  ⚠️   Some checks failed — review output above.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
