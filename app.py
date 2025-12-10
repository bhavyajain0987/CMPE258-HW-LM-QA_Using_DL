"""
QA Model Comparator - Main Entry Point

This application compares DistilBERT vs RoBERTa
on question answering tasks using custom text or standard benchmarks.

Modular Structure:
- config.py: Device detection and global state
- samples.py: Predefined test samples
- models.py: Model loading functionality
- evaluation.py: Evaluation and comparison logic
- ui.py: Gradio UI layout
"""

from ui import create_ui
import config

print(f"Running on: {config.device.upper()}")

# Create and launch the UI
demo = create_ui()
demo.launch()