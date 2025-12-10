"""
Configuration and global state management for the QA Model Comparator.
"""
import torch

# Device setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running on: {device.upper()}")

# Global state - models loaded lazily after UI launches
qa_fast = None
qa_accurate = None
metric = None
bertscore_metric = None
rouge_metric = None
squad_subset = None
models_ready = False
