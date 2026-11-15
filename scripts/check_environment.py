"""
check_environment.py
--------------------
Reusable GPU and package compatibility checker for the
Knowledge Distillation + Quantization project.

Usage:
    python scripts/check_environment.py

Compatible with Python 3.8+ and Google Colab.
Does NOT download any models or datasets.
"""

import sys
import subprocess
import importlib
import platform
from pathlib import Path

# ── formatting helpers ──────────────────────────────────────────────────────
PASS  = "  [PASS]"
FAIL  = "  [FAIL]"
WARN  = "  [WARN]"
INFO  = "  [INFO]"
SEP   = "─" * 64


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ── 1. Python version ───────────────────────────────────────────────────────
def check_python() -> dict:
    section("1. Python Interpreter")
    info = {
        "version":    platform.python_version(),
        "executable": sys.executable,
        "prefix":     sys.prefix,
        "platform":   platform.platform(),
    }
    major, minor = sys.version_info[:2]
    print(f"{INFO}  Version    : {info['version']}")
    print(f"{INFO}  Executable : {info['executable']}")
    print(f"{INFO}  Prefix     : {info['prefix']}")
    print(f"{INFO}  OS         : {info['platform']}")

    if (major, minor) < (3, 8):
        print(f"{FAIL}  Python 3.8+ required.")
        info["ok"] = False
    elif (major, minor) >= (3, 13):
        print(f"{WARN}  Python {major}.{minor} is AHEAD of the tested range (3.8–3.12).")
        print(f"{WARN}  HuggingFace Transformers and several wheels do NOT yet publish")
        print(f"{WARN}  Python 3.13/3.14 builds. Migration to Python 3.11 is recommended.")
        info["ok"] = False
        info["migration_needed"] = True
    else:
        print(f"{PASS}  Python {major}.{minor} is within the supported range (3.8–3.12).")
        info["ok"] = True
    return info


# ── 2. NVIDIA GPU via nvidia-smi ────────────────────────────────────────────
def check_nvidia_smi() -> dict:
    section("2. NVIDIA GPU (nvidia-smi)")
    result = {"available": False}
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            timeout=10,
        ).decode().strip()

        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 4:
            name, total_mb, free_mb, driver = parts[0], parts[1], parts[2], parts[3]
        else:
            name, total_mb, free_mb, driver = out, "?", "?", "?"

        # Grab CUDA version from full nvidia-smi output
        full = subprocess.check_output(
            ["nvidia-smi"], stderr=subprocess.STDOUT, timeout=10
        ).decode()
        cuda_ver = "?"
        for line in full.splitlines():
            if "CUDA Version" in line:
                cuda_ver = line.split("CUDA Version:")[-1].strip().split()[0]
                break

        result.update({
            "available": True,
            "name":      name,
            "total_mb":  total_mb,
            "free_mb":   free_mb,
            "driver":    driver,
            "cuda_ver":  cuda_ver,
        })
        print(f"{PASS}  GPU model      : {name}")
        print(f"{INFO}  VRAM total     : {total_mb} MiB")
        print(f"{INFO}  VRAM free      : {free_mb} MiB")
        print(f"{INFO}  Driver version : {driver}")
        print(f"{INFO}  CUDA version   : {cuda_ver}")
    except FileNotFoundError:
        print(f"{FAIL}  nvidia-smi not found. No NVIDIA GPU detected or driver not installed.")
    except subprocess.CalledProcessError as e:
        print(f"{FAIL}  nvidia-smi error: {e.output.decode()[:200]}")
    except Exception as e:
        print(f"{FAIL}  Unexpected error: {e}")
    return result


# ── 3. PyTorch CUDA build ───────────────────────────────────────────────────
def check_pytorch() -> dict:
    section("3. PyTorch Build")
    result = {"available": False, "cuda_enabled": False}
    try:
        import torch  # noqa: PLC0415
        result["available"]   = True
        result["version"]     = torch.__version__
        result["cuda_built"]  = torch.version.cuda
        result["cuda_avail"]  = torch.cuda.is_available()
        result["device_cnt"]  = torch.cuda.device_count()

        print(f"{INFO}  torch.__version__       : {torch.__version__}")
        print(f"{INFO}  torch.version.cuda      : {torch.version.cuda}")
        print(f"{INFO}  torch.cuda.is_available : {torch.cuda.is_available()}")
        print(f"{INFO}  torch.cuda.device_count : {torch.cuda.device_count()}")

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // 1024**2
            result["gpu_name"] = name
            result["vram_mb"]  = vram
            result["cuda_enabled"] = True
            print(f"{PASS}  GPU detected    : {name}")
            print(f"{INFO}  VRAM (torch)    : {vram} MiB")
        else:
            if torch.version.cuda is None:
                print(f"{FAIL}  Installed torch is the CPU-ONLY build (+cpu suffix).")
                print(f"{WARN}  CUDA will NOT be used even though an NVIDIA GPU is present.")
            else:
                print(f"{WARN}  torch was built with CUDA {torch.version.cuda}")
                print(f"{WARN}  but torch.cuda.is_available() == False.")
                print(f"{WARN}  Possible driver/runtime version mismatch.")
    except ImportError:
        print(f"{FAIL}  torch not installed.")
    return result


# ── 4. Package inventory ─────────────────────────────────────────────────────
def check_packages() -> dict:
    section("4. Required Packages")
    packages = {
        "torch":           "PyTorch",
        "transformers":    "HuggingFace Transformers",
        "datasets":        "HuggingFace Datasets",
        "evaluate":        "HuggingFace Evaluate",
        "accelerate":      "HuggingFace Accelerate",
        "tokenizers":      "HuggingFace Tokenizers",
        "huggingface_hub": "HuggingFace Hub",
        "sklearn":         "scikit-learn",
        "numpy":           "NumPy",
        "pandas":          "Pandas",
        "matplotlib":      "Matplotlib",
        "streamlit":       "Streamlit",
        "tqdm":            "tqdm",
    }
    found, missing = {}, []
    for pkg, label in packages.items():
        try:
            m = importlib.import_module(pkg)
            v = getattr(m, "__version__", "?")
            found[pkg] = v
            print(f"{PASS}  {label:<35} {v}")
        except ImportError:
            missing.append(pkg)
            print(f"{FAIL}  {label:<35} NOT INSTALLED")
    return {"found": found, "missing": missing, "ok": len(missing) == 0}


# ── 5. VRAM-based training recommendations ──────────────────────────────────
def make_recommendations(gpu: dict, torch_info: dict, python_info: dict) -> dict:
    section("5. Training Recommendations")

    cuda_ok   = torch_info.get("cuda_enabled", False)
    vram_mb   = torch_info.get("vram_mb", 0) if cuda_ok else int(gpu.get("total_mb", 0) or 0)
    gpu_name  = gpu.get("name", "Unknown")

    recs = {}

    if vram_mb >= 16_000:
        tier = "HIGH (16 GB+)"
        bert_local     = True
        distil_local   = True
        batch_size     = 32
        mixed_prec     = True
        grad_accum     = False
        grad_accum_steps = 1
        colab_needed   = False
    elif vram_mb >= 8_000:
        tier = "MID (8–16 GB)"
        bert_local     = True
        distil_local   = True
        batch_size     = 16
        mixed_prec     = True
        grad_accum     = False
        grad_accum_steps = 1
        colab_needed   = False
    elif vram_mb >= 4_000:
        tier = "LOW (4–8 GB)"
        bert_local     = False   # BERT-base FP32 peaks ~6–8 GB at bs=16
        distil_local   = True
        batch_size     = 8
        mixed_prec     = True    # fp16 halves VRAM usage
        grad_accum     = True
        grad_accum_steps = 4     # effective batch = 32
        colab_needed   = True    # for BERT teacher training
    elif vram_mb > 0:
        tier = "VERY LOW (<4 GB)"
        bert_local     = False
        distil_local   = False
        batch_size     = 4
        mixed_prec     = True
        grad_accum     = True
        grad_accum_steps = 8
        colab_needed   = True
    else:
        tier = "CPU ONLY"
        bert_local     = False
        distil_local   = False
        batch_size     = 8
        mixed_prec     = False
        grad_accum     = True
        grad_accum_steps = 4
        colab_needed   = True

    recs = {
        "gpu":                gpu_name,
        "vram_mb":            vram_mb,
        "tier":               tier,
        "bert_local":         bert_local,
        "distilbert_local":   distil_local,
        "batch_size":         batch_size,
        "mixed_precision":    mixed_prec,
        "grad_accum":         grad_accum,
        "grad_accum_steps":   grad_accum_steps,
        "colab_preferred":    colab_needed,
    }

    print(f"{INFO}  GPU              : {gpu_name}")
    print(f"{INFO}  VRAM tier        : {tier}  ({vram_mb} MiB)")
    print(f"{INFO}  CUDA usable      : {cuda_ok}")
    print()
    _yn = lambda b: "YES" if b else "NO"
    print(f"{INFO}  BERT-base local training   : {_yn(bert_local)}")
    print(f"{INFO}  DistilBERT local training  : {_yn(distil_local)}")
    print(f"{INFO}  Recommended batch_size     : {batch_size}")
    print(f"{INFO}  Mixed precision (fp16/bf16): {_yn(mixed_prec)}")
    print(f"{INFO}  Gradient accumulation      : {_yn(grad_accum)}"
          + (f"  (steps={grad_accum_steps})" if grad_accum else ""))

    if not cuda_ok and vram_mb > 0:
        print()
        print(f"{WARN}  *** GPU is present but PyTorch cannot use it. ***")
        print(f"{WARN}  The CPU-only PyTorch build must be replaced with a CUDA build.")
        print(f"{WARN}  See migration guide in the Summary section below.")

    return recs


# ── 6. Migration guidance ───────────────────────────────────────────────────
def print_migration_guide(python_info: dict, torch_info: dict, gpu: dict) -> None:
    section("6. Migration / Fix Guide")

    need_python = python_info.get("migration_needed", False)
    need_cuda   = not torch_info.get("cuda_enabled", False) and gpu.get("available", False)
    need_hf     = True  # transformers/datasets are always missing on this env

    if need_python:
        print("""  ── Python 3.11 Migration (REQUIRED) ────────────────────────
  Your current interpreter is Python 3.14.2.
  HuggingFace Transformers, Datasets, and most ML wheels only
  publish pre-built binaries for Python 3.8–3.12.

  Step 1 — Download and install Python 3.11 (keep 3.14 intact):
      https://www.python.org/downloads/release/python-3119/
      Choose: Windows installer (64-bit)
      ✔ Add python.exe to PATH  (check the box)
      ✔ Install for all users is optional

  Step 2 — Create a dedicated virtual environment with Python 3.11:
      py -3.11 -m venv .venv311
      .venv311\\Scripts\\Activate.ps1

  Step 3 — Confirm the active interpreter:
      python --version          # should print Python 3.11.x
      python -m pip --version
""")

    if need_cuda:
        print("""  ── CUDA-enabled PyTorch (REQUIRED for GPU training) ────────
  Your GPU (RTX 4050, VRAM ~6 GB) supports CUDA.
  Your NVIDIA driver (581.86) reports CUDA 13.0, which is
  backward-compatible with CUDA toolkit 12.4.

  After activating the Python 3.11 venv, install the CUDA 12.4 build:

      pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \\
          --index-url https://download.pytorch.org/whl/cu124

  Verify GPU access afterwards:
      python -c "import torch; print(torch.cuda.is_available())"
      # expected: True
""")

    if need_hf:
        print("""  ── HuggingFace + Project Packages ───────────────────────────
  After the Python and PyTorch steps above, install everything else:

      pip install -r requirements.txt

  If torch was already installed via the CUDA wheel above, pip will
  skip re-installing it (version already satisfies the constraint).
""")

    if not need_python and not need_cuda:
        print(f"{PASS}  No critical migration steps needed.")


# ── 7. Final verdict ─────────────────────────────────────────────────────────
def print_verdict(python_info: dict, torch_info: dict, pkg_info: dict, recs: dict) -> None:
    section("7. Final Verdict")

    cuda_ok      = torch_info.get("cuda_enabled", False)
    py_ok        = python_info.get("ok", False)
    pkgs_ok      = pkg_info.get("ok", False)
    colab_pref   = recs.get("colab_preferred", True)
    vram_mb      = recs.get("vram_mb", 0)

    if py_ok and cuda_ok and pkgs_ok and not colab_pref:
        verdict = "LOCAL TRAINING"
        detail  = "Environment is fully configured. Proceed with Phase 2B."
    elif not py_ok or not cuda_ok:
        verdict = "MIGRATION REQUIRED → then LOCAL TRAINING"
        detail  = (
            "Python 3.14 and/or CPU-only PyTorch must be fixed first.\n"
            "  After migration your RTX 4050 (6 GB VRAM) can handle DistilBERT\n"
            "  training locally. For BERT-base teacher fine-tuning (needs ~6–8 GB\n"
            "  peak), use Google Colab (free T4 = 15 GB) or enable fp16 + gradient\n"
            "  accumulation to stay within 6 GB."
        )
    elif colab_pref:
        verdict = "GOOGLE COLAB RECOMMENDED"
        detail  = (
            f"GPU VRAM is {vram_mb} MiB. DistilBERT fits, but BERT-base teacher\n"
            "  fine-tuning is tight. Use Colab for teacher training (Phase 3) and\n"
            "  local GPU for student + quantization phases."
        )
    else:
        verdict = "LOCAL TRAINING (with caveats)"
        detail  = "Some packages are missing. Install requirements.txt first."

    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │  VERDICT:  {verdict:<47}│")
    print(f"  └─────────────────────────────────────────────────────────┘")
    print(f"\n  {detail}\n")

    if recs:
        print("  Recommended config.py adjustments after migration:")
        print(f"    BATCH_SIZE          = {recs['batch_size']}")
        print(f"    MIXED_PRECISION     = {recs['mixed_precision']}")
        if recs["grad_accum"]:
            print(f"    GRADIENT_ACCUM_STEPS= {recs['grad_accum_steps']}")
        print()


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("\n" + "=" * 64)
    print("  Knowledge Distillation + Quantization — Environment Check")
    print("=" * 64)

    python_info = check_python()
    gpu_info    = check_nvidia_smi()
    torch_info  = check_pytorch()
    pkg_info    = check_packages()
    recs        = make_recommendations(gpu_info, torch_info, python_info)
    print_migration_guide(python_info, torch_info, gpu_info)
    print_verdict(python_info, torch_info, pkg_info, recs)
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
