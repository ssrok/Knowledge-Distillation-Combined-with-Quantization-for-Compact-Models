"""
make_colab_zip.py
-----------------
Creates colab_upload.zip in the project root containing the files needed to
run the dataset preparation notebook on Google Colab:

    src/              — project source package
    notebooks/        — all project notebooks
    requirements.txt  — dependency list

Usage (run from anywhere inside the project):
    python scripts/make_colab_zip.py

The resulting colab_upload.zip is placed in the project root.
Upload it when Colab's file picker appears at Step 3 of the notebook.
"""

import sys
import zipfile
from pathlib import Path


def make_zip(output_path: Path, project_root: Path) -> None:
    """Write a zip archive containing the Colab-required files."""

    # Files and directories to include
    includes = [
        project_root / "src",
        project_root / "notebooks",
        project_root / "requirements.txt",
    ]

    # Files and patterns to exclude inside those directories
    exclude_dirs  = {"__pycache__", ".ipynb_checkpoints", ".venv311", ".venv"}
    exclude_exts  = {".pyc", ".pyo", ".pyd"}

    if output_path.exists():
        output_path.unlink()
        print(f"Removed existing {output_path.name}")

    added = []
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in includes:
            if not item.exists():
                print(f"  WARNING: {item} not found — skipping")
                continue

            if item.is_file():
                arcname = item.relative_to(project_root)
                zf.write(item, arcname)
                added.append(str(arcname))
            else:
                # Directory: walk and add all matching files
                for path in sorted(item.rglob("*")):
                    # Skip excluded directories
                    if any(part in exclude_dirs for part in path.parts):
                        continue
                    # Skip excluded extensions
                    if path.suffix in exclude_exts:
                        continue
                    if path.is_file():
                        arcname = path.relative_to(project_root)
                        zf.write(path, arcname)
                        added.append(str(arcname))

    size_kb = output_path.stat().st_size / 1024
    print(f"\nCreated: {output_path}")
    print(f"Size   : {size_kb:.1f} KB")
    print(f"Files  : {len(added)}")
    print("\nContents:")
    for f in added:
        print(f"  {f}")


def main() -> None:
    # Resolve project root from this script's location (scripts/ is one level in)
    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent
    output_path  = project_root / "colab_upload.zip"

    print("=" * 55)
    print("  Building colab_upload.zip")
    print("=" * 55)
    print(f"Project root : {project_root}")
    print(f"Output       : {output_path}\n")

    make_zip(output_path, project_root)

    print("\n" + "=" * 55)
    print("  Done. Upload colab_upload.zip when Colab")
    print("  prompts for a file at Step 3 of the notebook.")
    print("=" * 55)


if __name__ == "__main__":
    main()
