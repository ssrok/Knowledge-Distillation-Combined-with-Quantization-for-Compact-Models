# Knowledge Distillation Combined with Quantization for Compact Models

A research prototype that studies the trade-off between model size, inference
speed, and classification accuracy by combining **knowledge distillation** and
**dynamic INT8 quantization** applied to Transformer models on the SST-2
sentiment-analysis benchmark.

---

## Experimental Models

| ID | Label | Architecture | Precision | Training strategy |
|----|-------|-------------|-----------|-------------------|
| M1 | Teacher | BERT-base-uncased | FP32 | Fine-tuned on SST-2 |
| M2 | Student baseline | DistilBERT-base-uncased | FP32 | Fine-tuned directly on SST-2 |
| M3 | Distilled student | DistilBERT-base-uncased | FP32 | Knowledge distillation from M1 |
| M4 | Quantized baseline | DistilBERT-base-uncased | INT8 | M2 + dynamic quantization |
| M5 | Quantized distilled | DistilBERT-base-uncased | INT8 | M3 + dynamic quantization |

**Research questions**
- Does knowledge distillation recover accuracy lost when switching from BERT to
  DistilBERT?
- Does INT8 quantization further compress models with minimal accuracy penalty?
- What is the cumulative effect of combining distillation and quantization on
  model size and inference throughput?

---

## Project Structure

```
knowledge-distillation-quantization/
├── data/                        # Downloaded / preprocessed datasets (git-ignored)
├── models/
│   ├── teacher/                 # M1 checkpoint
│   ├── student_baseline/        # M2 checkpoint
│   ├── student_distilled/       # M3 checkpoint
│   └── student_quantized/       # M4 & M5 serialised models
├── notebooks/
│   ├── 01_teacher_training.ipynb
│   ├── 02_student_baseline.ipynb
│   ├── 03_knowledge_distillation.ipynb
│   ├── 04_quantization.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py                # Central hyper-parameters and paths
│   ├── dataset.py               # Data loading and tokenisation
│   ├── teacher.py               # M1 training/loading
│   ├── student.py               # M2 training/loading
│   ├── distillation.py          # M3 distillation loss and training
│   ├── quantization.py          # M4/M5 quantisation helpers
│   ├── evaluation.py            # Unified metrics and plots
│   └── utils.py                 # Seeds, logging, checkpoint helpers
├── results/                     # Generated tables and figures (git-ignored)
├── app/
│   └── streamlit_app.py         # Interactive demo
├── requirements.txt
├── .gitignore
├── setup_verify.py              # Lightweight environment validation script
└── README.md
```

---

## Setup

### 1 — Clone
```bash
git clone <repo-url>
cd knowledge-distillation-quantization
```

### 2 — Create a virtual environment (local)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Verify the environment
```bash
python setup_verify.py
```

### Google Colab
Mount your Drive or clone directly in a cell, then run:
```python
!pip install -r requirements.txt
!python setup_verify.py
```
Training notebooks are GPU-ready (CUDA detected automatically via `config.DEVICE`).

---

## Implementation Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Project initialisation, config, placeholders, README | ✅ Complete |
| 2 | Dataset loading, tokenisation, DataLoaders | 🔜 Next |
| 3 | M1 teacher fine-tuning; M2 student baseline | 🔜 Planned |
| 4 | M3 knowledge distillation training loop | 🔜 Planned |
| 5 | M4/M5 dynamic INT8 quantization | 🔜 Planned |
| 6 | Unified evaluation, comparison tables, plots | 🔜 Planned |
| 7 | Streamlit demo app | 🔜 Planned |

---

## Key Hyper-parameters (see `src/config.py`)

| Parameter | Value |
|-----------|-------|
| Teacher model | bert-base-uncased |
| Student model | distilbert-base-uncased |
| Dataset | GLUE SST-2 |
| Max sequence length | 128 |
| Batch size | 16 |
| Epochs | 2 (configurable to 3) |
| Learning rate | 2e-5 |
| Distillation temperature | 2.0 |
| Distillation alpha | 0.5 |
| Random seed | 42 |

---

## Citation / References

- Hinton et al., *Distilling the Knowledge in a Neural Network*, 2015.
- Sanh et al., *DistilBERT, a distilled version of BERT*, 2019.
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers*, 2018.
- PyTorch Dynamic Quantization: https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html
- Hugging Face Transformers: https://huggingface.co/docs/transformers
