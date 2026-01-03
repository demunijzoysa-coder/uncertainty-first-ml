# Uncertainty-First ML: Self-Aware Decision Engine

A classifier that not only predicts, but estimates uncertainty, calibrates confidence, and abstains ("I don't know") when reliability is low—tested under distribution shift (OOD).

## Why this exists
Most models always answer, even when wrong. This project builds a decision engine that:
- quantifies uncertainty (not just confidence)
- calibrates predictions
- refuses to answer when risk is high
- exposes failure cases under distribution shift

## Methods (roadmap)
- [ ] Baseline classifier (CIFAR-10)
- [ ] Uncertainty: MC Dropout
- [ ] Uncertainty: Deep Ensembles
- [ ] Calibration: Reliability diagrams + ECE + Temperature Scaling
- [ ] Abstention: coverage vs accuracy + risk-coverage curves
- [ ] OOD testing (e.g., SVHN / CIFAR-100)
- [ ] Dashboard demo (Streamlit/Gradio)
- [ ] Paper-style report + reproducibility checklist

## Results (will be filled as experiments complete)
| Model | ID Acc | ECE ↓ | NLL ↓ | OOD AUROC ↑ | Coverage@95%Acc ↑ |
|------|--------|-------|-------|-------------|-------------------|
| Baseline | - | - | - | - | - |
| MC Dropout | - | - | - | - | - |
| Deep Ensemble | - | - | - | - | - |

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python -m src.train --config experiments/configs/baseline.yaml
