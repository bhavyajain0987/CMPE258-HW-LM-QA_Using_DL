"""
Model loading functionality.
"""
import gradio as gr
import evaluate
from transformers import pipeline
from datasets import load_dataset
import config


def load_all_models(progress=gr.Progress()):
    """Load all models and datasets with progress updates."""
    
    try:
        progress(0.1, desc="🔄 Loading DistilBERT (Fast model)...")
        config.qa_fast = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=config.device)
        
        progress(0.3, desc="🔄 Loading RoBERTa (Accurate model)...")
        config.qa_accurate = pipeline("question-answering", model="deepset/roberta-base-squad2", device=config.device)
        
        progress(0.5, desc="🔄 Loading SQuAD dataset...")
        config.squad_subset = load_dataset("squad_v2", split="validation").shuffle(seed=42).select(range(50))
        
        progress(0.7, desc="🔄 Loading SQuAD metric...")
        config.metric = evaluate.load("squad_v2")
        
        progress(0.8, desc="🔄 Loading BERTScore metric...")
        config.bertscore_metric = evaluate.load("bertscore")
        
        progress(0.9, desc="🔄 Loading ROUGE metric...")
        config.rouge_metric = evaluate.load("rouge")
        
        progress(1.0, desc="✅ All models loaded!")
        config.models_ready = True
        
        return (
            """<div style="padding: 20px; background: linear-gradient(90deg, #1a472a, #2d5a3d); border-radius: 12px; border: 2px solid #4ade80; margin: 10px 0;">
                <h2 style="margin: 0; color: #4ade80; font-size: 24px;">✅ Models Loaded Successfully!</h2>
                <p style="margin: 8px 0 0 0; color: #86efac; font-size: 16px;">You can now use all features.</p>
            </div>""",
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.Button(interactive=True)
        )
    except Exception as e:
        return (
            f"""<div style="padding: 20px; background: linear-gradient(90deg, #4a1a1a, #5a2d2d); border-radius: 12px; border: 2px solid #f87171; margin: 10px 0;">
                <h2 style="margin: 0; color: #f87171; font-size: 24px;">❌ Error Loading Models</h2>
                <p style="margin: 8px 0 0 0; color: #fca5a5; font-size: 16px;">{str(e)}</p>
            </div>""",
            gr.Button(interactive=False),
            gr.Button(interactive=False),
            gr.Button(interactive=False)
        )
