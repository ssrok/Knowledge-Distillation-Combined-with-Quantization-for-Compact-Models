"""
dataset.py
----------
SST-2 dataset pipeline for the Knowledge Distillation + Quantization project.

Public API
----------
load_sst2()                 — download / load raw GLUE SST-2 splits
inspect_dataset()           — print structure and class distribution
tokenize_dataset()          — apply BERT tokenizer, keep labels & idx
export_processed()          — save Parquet + JSON metadata (CSV fallback)
load_processed_local()      — reload from Parquet/CSV without HF datasets
get_dataloaders()           — return PyTorch DataLoaders (train, val)
validate_dataset()          — run integrity checks and print a summary
run_pipeline()              — convenience: end-to-end from download to export

Design notes
------------
* The official GLUE SST-2 test split has no ground-truth labels; it is
  NEVER used for evaluation.  The official `validation` split (872 examples,
  labelled) is the held-out evaluation set for all five models.

* Exported files store raw sentences, labels, and IDs — NOT padded token
  sequences — so the files are compact and re-tokenisable with any tokenizer.
  Token sequences are computed at DataLoader creation time via a DataCollator.

* All five model configurations (M1–M5) load from the same exported files to
  guarantee identical data partitions.

* Subset mode: set config.SUBSET_SIZE to a positive int (e.g. 500) for fast
  debugging.  Uses a seeded stratified sample so the class balance is preserved.

Colab compatibility
-------------------
This module uses `datasets` and `transformers` from Hugging Face.  On the
local Windows machine these are blocked by the WDAC policy for pyarrow-
dependent packages.  Run the export step (run_pipeline / export_processed)
on Google Colab, then copy data/processed/ back to the local project before
local training.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_config():
    """Lazy import of config to avoid circular imports."""
    from src import config  # noqa: PLC0415
    return config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: PLC0415
        torch.manual_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 1. Load raw dataset
# ---------------------------------------------------------------------------

def load_sst2(subset_size: Optional[int] = None, seed: int = 42):
    """Load GLUE SST-2 from the Hugging Face Hub.

    Args:
        subset_size: If set to a positive integer, return only this many
                     examples per split (stratified by label, seeded).
                     Use None for the full dataset.
        seed:        Random seed used for subsampling.

    Returns:
        datasets.DatasetDict with keys 'train' and 'validation'.
        The official 'test' split is excluded because it has no labels.

    Notes:
        * Train  : 67,349 examples
        * Validation: 872 examples (labelled, used for evaluation)
    """
    from datasets import load_dataset  # noqa: PLC0415

    cfg = _get_config()
    _set_seed(seed)

    logger.info("Loading GLUE SST-2 …")
    raw = load_dataset(cfg.DATASET_NAME, cfg.DATASET_CONFIG)

    # Drop test split (no ground-truth labels)
    splits = {k: raw[k] for k in ("train", "validation")}

    if subset_size is not None and subset_size > 0:
        logger.info("Subset mode: %d examples per split", subset_size)
        splits = {
            k: _stratified_sample(v, subset_size, seed)
            for k, v in splits.items()
        }

    from datasets import DatasetDict  # noqa: PLC0415
    dataset = DatasetDict(splits)
    logger.info(
        "Loaded — train: %d  val: %d",
        len(dataset["train"]),
        len(dataset["validation"]),
    )
    return dataset


def _stratified_sample(split, n: int, seed: int):
    """Return a seeded stratified subsample of `split` with at most n examples."""
    import pandas as pd  # noqa: PLC0415
    df = split.to_pandas()
    n_per_class = max(1, n // 2)
    # include_groups=False avoids pandas 2.x DeprecationWarning
    sampled = (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), n_per_class), random_state=seed),
               include_groups=False)
        .reset_index(drop=True)
    )
    # restore the 'label' column that groupby dropped with include_groups=False
    if "label" not in sampled.columns:
        sampled = df.loc[sampled.index].reset_index(drop=True)
    from datasets import Dataset  # noqa: PLC0415
    return Dataset.from_pandas(sampled, preserve_index=False)


# ---------------------------------------------------------------------------
# 2. Inspect dataset
# ---------------------------------------------------------------------------

def inspect_dataset(dataset) -> Dict:
    """Print and return a summary of the dataset.

    Args:
        dataset: DatasetDict returned by load_sst2().

    Returns:
        dict with keys per split containing count, label_dist, columns.
    """
    cfg = _get_config()
    summary = {}

    print("\n" + "=" * 60)
    print("  Dataset Inspection: GLUE SST-2")
    print("=" * 60)
    print(f"  Splits  : {list(dataset.keys())}")
    print(f"  Columns : {dataset['train'].column_names}")
    print(f"  Features: {dataset['train'].features}")

    for split_name, split_data in dataset.items():
        import pandas as pd  # noqa: PLC0415
        df = split_data.to_pandas()

        label_dist = df[cfg.LABEL_COLUMN].value_counts().sort_index().to_dict()
        label_pct  = {
            cfg.LABEL_NAMES.get(k, str(k)): round(v / len(df) * 100, 1)
            for k, v in label_dist.items()
        }

        summary[split_name] = {
            "count":      len(df),
            "label_dist": label_dist,
            "label_pct":  label_pct,
            "columns":    list(df.columns),
        }

        print(f"\n  Split: {split_name}")
        print(f"    Examples : {len(df):,}")
        print(f"    Label 0 (negative): {label_dist.get(0, 0):,}  "
              f"({label_pct.get('negative', 0)}%)")
        print(f"    Label 1 (positive): {label_dist.get(1, 0):,}  "
              f"({label_pct.get('positive', 0)}%)")
        print(f"    Sample sentences:")
        for row in df.sample(3, random_state=42).itertuples():
            print(f"      [{row.label}] {row.sentence[:80]}")

    print("=" * 60 + "\n")
    return summary


# ---------------------------------------------------------------------------
# 3. Tokenise
# ---------------------------------------------------------------------------

def get_tokenizer(model_name: Optional[str] = None):
    """Load and return the BERT-base-uncased tokenizer.

    Args:
        model_name: HuggingFace model name.  Defaults to config.TEACHER_MODEL_NAME.

    Returns:
        transformers.AutoTokenizer instance.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415
    cfg = _get_config()
    name = model_name or cfg.TEACHER_MODEL_NAME
    logger.info("Loading tokenizer: %s", name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    return tokenizer


def tokenize_dataset(dataset, tokenizer, max_length: Optional[int] = None):
    """Tokenise every sentence in `dataset`.

    Tokenisation is done WITHOUT padding so that sequences retain their
    natural lengths.  Padding to the batch maximum is applied at collation
    time inside the DataLoader via DataCollatorWithPadding.

    Args:
        dataset:    DatasetDict from load_sst2().
        tokenizer:  Tokenizer from get_tokenizer().
        max_length: Max token length (defaults to config.MAX_SEQ_LENGTH).

    Returns:
        DatasetDict with added columns:
            input_ids, attention_mask  (variable-length lists)
        Retained columns:
            sentence, label, idx
    """
    cfg = _get_config()
    max_len = max_length or cfg.MAX_SEQ_LENGTH

    def _tokenize_batch(batch):
        return tokenizer(
            batch[cfg.TEXT_COLUMN],
            truncation=True,
            max_length=max_len,
            padding=False,   # dynamic padding at collation time
        )

    logger.info("Tokenising dataset (max_length=%d) …", max_len)
    tokenized = dataset.map(
        _tokenize_batch,
        batched=True,
        desc="Tokenising",
    )

    # Keep only the columns needed downstream
    keep = ["input_ids", "attention_mask", "label", "idx"]
    tokenized = tokenized.remove_columns(
        [c for c in tokenized["train"].column_names if c not in keep]
    )
    tokenized = tokenized.rename_column("label", "labels")

    # Set format so the DataLoader receives tensors directly
    tokenized.set_format("torch")

    logger.info("Tokenisation complete.")
    return tokenized


# ---------------------------------------------------------------------------
# 4. Export processed data
# ---------------------------------------------------------------------------

def export_processed(
    dataset,
    output_dir: Optional[Path] = None,
    split_meta: Optional[Dict] = None,
) -> Dict[str, Path]:
    """Save portable processed datasets and metadata.

    Exported files store the raw sentence text, label, and idx — NOT padded
    token sequences — so the files are compact and can be reloaded without
    the HuggingFace datasets library.

    Format preference: Parquet.  Falls back to CSV if pyarrow is unavailable.

    Args:
        dataset:    DatasetDict from load_sst2() (raw, not tokenised).
        output_dir: Directory to write files into.
                    Defaults to config.PROCESSED_DIR.
        split_meta: Optional dict of summary statistics to embed in metadata.

    Returns:
        dict mapping 'train_file', 'val_file', 'metadata_file' to Path objects.
    """
    import pandas as pd  # noqa: PLC0415

    cfg   = _get_config()
    odir  = Path(output_dir) if output_dir else cfg.PROCESSED_DIR
    odir.mkdir(parents=True, exist_ok=True)

    cfg_meta = {
        "dataset_name":    cfg.DATASET_NAME,
        "dataset_config":  cfg.DATASET_CONFIG,
        "text_column":     cfg.TEXT_COLUMN,
        "label_column":    cfg.LABEL_COLUMN,
        "id_column":       cfg.ID_COLUMN,
        "num_labels":      cfg.NUM_LABELS,
        # JSON keys must be strings; convert int keys here
        "label_names":     {str(k): v for k, v in cfg.LABEL_NAMES.items()},
        "max_seq_length":  cfg.MAX_SEQ_LENGTH,
        "random_seed":     cfg.RANDOM_SEED,
        "tokenizer_name":  cfg.TEACHER_MODEL_NAME,
        "note": (
            "Exported files contain raw sentences, labels, and example IDs. "
            "Token sequences are NOT stored here — they are computed at "
            "DataLoader creation time to allow dynamic padding. "
            "The official GLUE SST-2 test split (unlabelled) is excluded."
        ),
    }

    paths: Dict[str, Path] = {}
    split_counts: Dict[str, int] = {}

    for split_name, hf_split in dataset.items():
        df = hf_split.to_pandas()

        # Canonical column order
        df = df[["idx", "sentence", "label"]].copy()
        df["split"] = split_name
        df = df.rename(columns={"idx": "example_id"})
        df["label_name"] = df["label"].map(cfg.LABEL_NAMES)

        split_counts[split_name] = len(df)

        # Try Parquet first
        parquet_path = odir / f"{split_name}.parquet"
        csv_path     = odir / f"{split_name}.csv"

        try:
            df.to_parquet(parquet_path, index=False)
            paths[f"{split_name}_file"] = parquet_path
            logger.info("Saved %s → %s  (%d rows)", split_name, parquet_path, len(df))
        except Exception as parquet_err:
            logger.warning(
                "Parquet unavailable (%s). Falling back to CSV.", parquet_err
            )
            df.to_csv(csv_path, index=False)
            paths[f"{split_name}_file"] = csv_path
            logger.info("Saved %s → %s  (%d rows)", split_name, csv_path, len(df))

    # Write metadata
    meta = {
        **cfg_meta,
        "splits": split_counts,
        **(split_meta or {}),
    }
    meta_path = odir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    paths["metadata_file"] = meta_path
    logger.info("Metadata saved → %s", meta_path)

    return paths


# ---------------------------------------------------------------------------
# 5. Load processed files locally (no HF datasets required)
# ---------------------------------------------------------------------------

def load_processed_local(
    processed_dir: Optional[Path] = None,
) -> Tuple[object, object, Dict]:
    """Load the exported Parquet (or CSV) files using only Pandas.

    This function does NOT require the HuggingFace datasets library and is
    therefore usable on the local Windows machine even when pyarrow / datasets
    imports are blocked by WDAC.

    Args:
        processed_dir: Directory containing the exported files.
                       Defaults to config.PROCESSED_DIR.

    Returns:
        (train_df, val_df, metadata_dict)
        DataFrames have columns: example_id, sentence, label, split, label_name
    """
    import pandas as pd  # noqa: PLC0415

    cfg  = _get_config()
    pdir = Path(processed_dir) if processed_dir else cfg.PROCESSED_DIR

    def _load_split(name: str):
        parquet = pdir / f"{name}.parquet"
        csv     = pdir / f"{name}.csv"
        if parquet.exists():
            return pd.read_parquet(parquet)
        elif csv.exists():
            return pd.read_csv(csv)
        else:
            raise FileNotFoundError(
                f"No processed {name} split found in {pdir}. "
                "Run export_processed() on Colab first."
            )

    train_df = _load_split("train")
    val_df   = _load_split("validation")

    meta_path = pdir / "metadata.json"
    metadata  = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    logger.info(
        "Loaded local processed data — train: %d  val: %d",
        len(train_df), len(val_df),
    )
    return train_df, val_df, metadata


# ---------------------------------------------------------------------------
# 6. PyTorch Dataset wrapper (works with pandas DataFrames)
# ---------------------------------------------------------------------------

class SST2Dataset:
    """Minimal PyTorch Dataset that tokenises on-the-fly from a DataFrame.

    This class is used for local training when processed CSV/Parquet files
    are available but the HuggingFace datasets library is blocked.  It applies
    the tokenizer in __getitem__ so the DataLoader can apply dynamic padding
    via DataCollatorWithPadding.

    Args:
        df:         DataFrame with 'sentence' and 'label' columns.
        tokenizer:  HuggingFace tokenizer.
        max_length: Maximum token sequence length.
    """

    def __init__(self, df, tokenizer, max_length: int = 128):
        import torch  # noqa: PLC0415
        self._sentences  = df["sentence"].tolist()
        self._labels     = df["label"].tolist()
        self._ids        = df["example_id"].tolist()
        self._tokenizer  = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._sentences)

    def __getitem__(self, idx: int) -> Dict:
        import torch  # noqa: PLC0415
        enc = self._tokenizer(
            self._sentences[idx],
            truncation=True,
            max_length=self._max_length,
            padding=False,
            return_tensors=None,  # return plain lists; collator handles tensors
        )
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels":         self._labels[idx],
        }


# ---------------------------------------------------------------------------
# 7. DataLoaders
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_source,
    tokenizer,
    batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
    num_workers: int = 0,
):
    """Build PyTorch DataLoaders for training and validation.

    Accepts either:
    * A tokenised HuggingFace DatasetDict (output of tokenize_dataset()), OR
    * A tuple (train_df, val_df) of Pandas DataFrames (from load_processed_local()).

    Dynamic padding is applied via DataCollatorWithPadding so sequences in
    each batch are padded only to the longest sequence in that batch, not to
    the global max_length.  This keeps memory usage lower.

    Args:
        data_source: HF DatasetDict or (train_df, val_df) tuple.
        tokenizer:   HuggingFace tokenizer (needed for DataCollatorWithPadding).
        batch_size:  Mini-batch size (defaults to config.BATCH_SIZE).
        max_length:  Max token length (defaults to config.MAX_SEQ_LENGTH).
        num_workers: DataLoader worker processes.

    Returns:
        (train_loader, val_loader) — both torch.utils.data.DataLoader instances.
    """
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415
    from transformers import DataCollatorWithPadding  # noqa: PLC0415

    cfg    = _get_config()
    bs     = batch_size or cfg.BATCH_SIZE
    maxlen = max_length or cfg.MAX_SEQ_LENGTH

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # Branch: HF DatasetDict vs Pandas DataFrames
    if hasattr(data_source, "keys"):
        # HuggingFace DatasetDict (already tokenised)
        train_ds = data_source["train"]
        val_ds   = data_source["validation"]
    else:
        # Tuple of DataFrames — wrap in SST2Dataset
        train_df, val_df = data_source
        train_ds = SST2Dataset(train_df, tokenizer, maxlen)
        val_ds   = SST2Dataset(val_df,   tokenizer, maxlen)

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        "DataLoaders created — train batches: %d  val batches: %d",
        len(train_loader),
        len(val_loader),
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# 8. Validation / integrity checks
# ---------------------------------------------------------------------------

def validate_dataset(dataset_or_dfs, source: str = "hf") -> Dict:
    """Run integrity checks and print a validation summary.

    Args:
        dataset_or_dfs: Either a HF DatasetDict or a (train_df, val_df) tuple.
        source:         'hf' or 'local' — controls which path to follow.

    Returns:
        dict with per-split validation results.
    """
    import pandas as pd  # noqa: PLC0415

    cfg = _get_config()

    if source == "hf":
        splits = {k: v.to_pandas() for k, v in dataset_or_dfs.items()}
    else:
        train_df, val_df = dataset_or_dfs
        splits = {"train": train_df, "validation": val_df}

    results = {}
    all_ok   = True

    print("\n" + "=" * 60)
    print("  Dataset Validation Report")
    print("=" * 60)

    for split_name, df in splits.items():
        issues: List[str] = []

        # Determine column names based on source
        id_col   = "example_id" if source == "local" else cfg.ID_COLUMN
        text_col = "sentence"
        lbl_col  = "label"

        # 1 — count
        n = len(df)

        # 2 — expected counts
        expected = {"train": 67349, "validation": 872}
        if cfg.SUBSET_SIZE:
            expected = {k: min(v, cfg.SUBSET_SIZE) for k, v in expected.items()}
        exp = expected.get(split_name)
        count_ok = (exp is None) or (abs(n - exp) <= 2)  # allow ±2 for rounding
        if not count_ok:
            issues.append(f"Expected ~{exp} rows, got {n}")

        # 3 — label distribution
        label_dist = df[lbl_col].value_counts().sort_index().to_dict()
        expected_labels = {0, 1}
        found_labels    = set(df[lbl_col].unique())
        if found_labels != expected_labels:
            issues.append(f"Unexpected labels: {found_labels}")

        # 4 — missing values
        null_counts = df[[id_col, text_col, lbl_col]].isnull().sum().to_dict()
        if any(v > 0 for v in null_counts.values()):
            issues.append(f"Missing values: {null_counts}")

        # 5 — duplicate IDs
        if id_col in df.columns:
            dup_ids = df[id_col].duplicated().sum()
            if dup_ids > 0:
                issues.append(f"{dup_ids} duplicate example IDs")

        # 6 — empty sentences
        empty = (df[text_col].str.strip() == "").sum()
        if empty > 0:
            issues.append(f"{empty} empty sentences")

        # 7 — label balance check (warn if >70/30)
        pos_frac = label_dist.get(1, 0) / n if n else 0
        if not (0.3 <= pos_frac <= 0.7):
            issues.append(f"Class imbalance: pos_frac={pos_frac:.2%}")

        status = "PASS" if not issues else "WARN"
        if status == "WARN":
            all_ok = False

        results[split_name] = {
            "status":     status,
            "count":      n,
            "label_dist": label_dist,
            "issues":     issues,
        }

        print(f"\n  Split: {split_name}")
        print(f"    [{status}] {n:,} examples")
        print(f"    Label distribution: {label_dist}")
        pct_pos = label_dist.get(1, 0) / n * 100 if n else 0
        print(f"    Positive: {pct_pos:.1f}%  Negative: {100-pct_pos:.1f}%")
        if issues:
            for issue in issues:
                print(f"    ⚠  {issue}")
        else:
            print("    No issues found.")

        # Sample sentences
        print("    Sample sentences:")
        sample = df.sample(min(3, n), random_state=42)
        for _, row in sample.iterrows():
            print(f"      [{'pos' if row[lbl_col] == 1 else 'neg'}] "
                  f"{str(row[text_col])[:85]}")

    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_ok else 'WARNINGS PRESENT'}")
    print("=" * 60 + "\n")

    return results


def validate_tokenization(tokenized_dataset, tokenizer, n_samples: int = 3) -> bool:
    """Spot-check tokenisation by decoding sample sequences.

    Args:
        tokenized_dataset: Output of tokenize_dataset().
        tokenizer:         The tokenizer used in tokenize_dataset().
        n_samples:         Number of samples to decode and display.

    Returns:
        True if all samples decoded without error.
    """
    print("\n" + "─" * 60)
    print("  Tokenisation Spot-Check")
    print("─" * 60)
    split = tokenized_dataset["train"]
    indices = random.sample(range(len(split)), min(n_samples, len(split)))

    all_ok = True
    for i in indices:
        try:
            sample   = split[i]
            ids      = sample["input_ids"]
            # ids may be a torch.Tensor when format="torch"; convert to list
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            decoded  = tokenizer.decode(ids, skip_special_tokens=False)
            label    = int(sample["labels"])
            seq_len  = len(ids)
            print(f"  idx={i}  len={seq_len}  label={label}")
            print(f"    decoded: {decoded[:100]}")
        except Exception as e:
            print(f"  ERROR decoding sample {i}: {e}")
            all_ok = False

    if all_ok:
        print("  All spot-checks passed.")
    print("─" * 60 + "\n")
    return all_ok


# ---------------------------------------------------------------------------
# 9. End-to-end convenience runner
# ---------------------------------------------------------------------------

def run_pipeline(
    subset_size: Optional[int] = None,
    output_dir:  Optional[Path] = None,
    seed:        int = 42,
    skip_export: bool = False,
) -> Dict:
    """Run the full dataset preparation pipeline.

    Steps
    -----
    1. Load SST-2 from HuggingFace Hub.
    2. Inspect and validate the raw dataset.
    3. Export raw sentences + labels to Parquet (CSV fallback).
    4. Tokenise and spot-check.

    Args:
        subset_size: Positive int to use a small subset; None for full data.
        output_dir:  Where to write processed files.
        seed:        Random seed.
        skip_export: Set True to skip file export (dry-run / inspection only).

    Returns:
        dict with 'dataset', 'tokenized', 'export_paths', 'validation'.
    """
    cfg = _get_config()
    _set_seed(seed)

    # Step 1 — load
    dataset = load_sst2(subset_size=subset_size, seed=seed)

    # Step 2 — inspect
    summary = inspect_dataset(dataset)

    # Step 3 — validate raw
    val_results = validate_dataset(dataset, source="hf")

    # Step 4 — export
    export_paths: Dict = {}
    if not skip_export:
        export_paths = export_processed(
            dataset,
            output_dir=output_dir,
            split_meta={"inspection_summary": summary},
        )

    # Step 5 — tokenise
    tokenizer = get_tokenizer()
    tokenized  = tokenize_dataset(dataset, tokenizer)

    # Step 6 — spot-check tokenisation
    validate_tokenization(tokenized, tokenizer)

    logger.info("Pipeline complete.")
    return {
        "dataset":      dataset,
        "tokenized":    tokenized,
        "export_paths": export_paths,
        "validation":   val_results,
        "tokenizer":    tokenizer,
    }
