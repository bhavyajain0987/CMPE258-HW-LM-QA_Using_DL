# Q&A Model Comparator (CMPE 258 HW)

Compare DistilBERT and RoBERTa on extractive question answering with a simple Gradio UI, predefined samples, and dataset benchmarks. Includes a Colab training notebook to demonstrate before/after improvements.

## Overview
- Two QA models: DistilBERT and RoBERTa (SQuAD2 fine-tuned).
- UI-first load: app launches immediately; models load in the background.
- Predefined samples to reveal nuanced differences (negation, distractors).
- Dataset benchmarking with multiple metrics: F1, Exact Match, BERTScore, ROUGE-L.
- Training notebook showing performance improvement after fine-tuning.

## Project Structure
- `app.py`: Entry point; launches the Gradio UI.
- `ui.py`: UI layout with tabs for Custom Test, Predefined Samples, Dataset Evaluation, Training.
- `models.py`: Lazy-loading of models, dataset slice, and metrics.
- `evaluation.py`: Logic for custom tests, sample comparison, and dataset evaluation.
- `samples.py`: Predefined contexts, questions, and expected answers.
- `config.py`: Device detection (Apple Silicon MPS/CPU) and global state.
- `Training_HW_LM_DL.ipynb`: Colab-friendly notebook for fine-tuning and metric comparison.

## Quick Start
Requirements (macOS, zsh):
```bash
# Recommended: use Python 3.10+
python3 -m venv .venv
source .venv/bin/activate

pip install gradio transformers datasets evaluate matplotlib pandas torch
# If using Apple Silicon, install a compatible torch build per PyTorch docs.
```
Run the app:
```bash
python3 app.py
```
Open the Gradio URL shown in the terminal.

## Using the App
- Tab 1: Custom Test — paste a context and ask a question; see both model answers and confidence.
- Tab 2: Predefined Samples — Some sample context and questions I have added to showcase the models; shows winner analysis.
- Tab 3: Dataset Evaluation — runs benchmarks on a SQuAD v2 validation slice; renders a metrics table and chart.
- Tab 4: Training — links to the notebook demonstrating fine-tuning and before/after comparison.

## Metrics
The app’s dataset evaluation computes:
- **SQuAD v2**: F1 and Exact Match.
- **BERTScore (F1)**: semantic similarity between predicted and gold answers.
- **ROUGE-L**: overlap-based similarity.

## Training Notebook (Excellent Rating)
- File: `Training_HW_LM_DL.ipynb`
- Installs dependencies, preprocesses a SQuAD slice, fine-tunes a QA head, and evaluates before/after.
- Produces a comparison plot with F1, EM, BERTScore, ROUGE-L.
- If widgets error in VS Code, either run in Colab or disable progress bars:
```python
import os
os.environ["TQDM_DISABLE"] = "1"
```

## Notes & Troubleshooting
- Apple Silicon: if MPS isn’t used automatically, the app falls back to CPU.
- First run downloads models/datasets; allow a few minutes.
- If BERTScore/ROUGE packages are missing, install:
```bash
pip install bert_score rouge_score
```
- VS Code notebook widgets: install `ipywidgets` or run in Colab.

## Screenshots / Report
- Add screenshots of: UI loading banner, predefined sample comparison, dataset chart, and notebook plot.
- Include the course assignment requirements image in your report.

## License
Academic use for CMPE 258 HW.
