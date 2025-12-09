import gradio as gr
import torch
import evaluate
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from datasets import load_dataset
import os

# 1. SETUP
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running on: {device.upper()}")

# Global state - models loaded lazily after UI launches
qa_fast = None
qa_accurate = None
metric = None
squad_subset = None
models_ready = False


def load_all_models(progress=gr.Progress()):
    """Load all models and datasets with progress updates."""
    global qa_fast, qa_accurate, metric, squad_subset, models_ready
    
    try:
        progress(0.1, desc="🔄 Loading DistilBERT (Fast model)...")
        qa_fast = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=device)
        
        progress(0.4, desc="🔄 Loading RoBERTa (Accurate model)...")
        qa_accurate = pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)
        
        progress(0.7, desc="🔄 Loading SQuAD dataset...")
        squad_subset = load_dataset("squad_v2", split="validation").shuffle(seed=42).select(range(50))
        
        progress(0.9, desc="🔄 Loading evaluation metric...")
        metric = evaluate.load("squad_v2")
        
        progress(1.0, desc="✅ All models loaded!")
        models_ready = True
        
        return (
            """<div style="padding: 20px; background: linear-gradient(90deg, #1a472a, #2d5a3d); border-radius: 12px; border: 2px solid #4ade80; margin: 10px 0;">
                <h2 style="margin: 0; color: #4ade80; font-size: 24px;">✅ Models Loaded Successfully!</h2>
                <p style="margin: 8px 0 0 0; color: #86efac; font-size: 16px;">You can now use all features.</p>
            </div>""",
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
            gr.Button(interactive=False)
        )


# --- Logic Functions ---

def run_custom_test(context, question):
    """Runs inference on a single custom sample."""
    if not models_ready:
        return "⏳ Models still loading...", "⏳ Please wait...", "Loading..."
    
    res1 = qa_fast(question=question, context=context)
    res2 = qa_accurate(question=question, context=context)
    
    # Format output for UI
    out1 = f"**Answer:** {res1['answer']}\n**Conf:** {res1['score']:.4f}"
    out2 = f"**Answer:** {res2['answer']}\n**Conf:** {res2['score']:.4f}"
    
    # Determine winner
    if res1['score'] > res2['score']:
        winner = "DistilBERT (Fast) is more confident."
    else:
        winner = "RoBERTa (Accurate) is more confident."
        
    return out1, out2, winner

def run_dataset_evaluation(progress=gr.Progress()):
    """Runs the comparison on the SQuAD subset with a progress bar."""
    if not models_ready:
        return pd.DataFrame(), None
    
    results_list = []
    models = [("DistilBERT", qa_fast), ("RoBERTa", qa_accurate)]
    
    progress(0, desc="Starting Evaluation...")
    
    for model_name, model_pipe in models:
        predictions = []
        references = []
        
        # Loop through dataset with progress update
        for i, item in enumerate(squad_subset):
            progress((i + 1) / len(squad_subset), desc=f"Testing {model_name}...")
            
            res = model_pipe(question=item['question'], context=item['context'])
            
            predictions.append({
                'id': item['id'], 
                'prediction_text': res['answer'],
                'no_answer_probability': 0.0
            })
            references.append({'id': item['id'], 'answers': item['answers']})
            
        # Compute metrics
        score = metric.compute(predictions=predictions, references=references)
        results_list.append({
            "Model": model_name, 
            "F1": round(score['f1'], 4), 
            "Exact Match": round(score['exact'], 4)
        })
    
    # Create Dataframe
    df = pd.DataFrame(results_list)
    
    # Generate Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    df.plot(kind='bar', x='Model', ax=ax, color=['#1f77b4', '#ff7f0e'])
    ax.set_title("Model Performance on SQuAD (Subset)")
    ax.set_ylabel("Score")
    plt.tight_layout()
    
    return df, fig

# --- UI Layout ---

with gr.Blocks(title="QA Model Assignment") as demo:
    gr.Markdown("# 🤖 Question Answering Model Comparator")
    gr.Markdown("Compare **DistilBERT** (Fast) vs **RoBERTa** (Accurate) on custom text or standard benchmarks.")
    
    # Loading status banner
    loading_status = gr.Markdown("""
        <div style="padding: 20px; background: linear-gradient(90deg, #1a3a5c, #2d4a6d); border-radius: 12px; border: 2px solid #60a5fa; margin: 10px 0;">
            <h2 style="margin: 0; color: #60a5fa; font-size: 24px;">⏳ Loading Models...</h2>
            <p style="margin: 8px 0 0 0; color: #93c5fd; font-size: 16px;">Please wait while the AI models are being loaded. This may take a moment.</p>
        </div>
    """)

    with gr.Tab("1. Custom Test"):
        gr.Markdown("### Test with your own text")
        with gr.Row():
            with gr.Column():
                ctx_input = gr.Textbox(lines=5, label="Context", placeholder="Paste a paragraph here...", value="The MacBook Air is a line of laptop computers developed and manufactured by Apple Inc. It consists of a full-size keyboard, a machined aluminum case, and, in the more modern structure, a thin light structure.")
                q_input = gr.Textbox(lines=2, label="Question", placeholder="Type a question...", value="Who manufactures the MacBook Air?")
                btn_test = gr.Button("Ask Models", variant="primary", interactive=False)
            with gr.Column():
                out_fast = gr.Textbox(label="DistilBERT Result")
                out_acc = gr.Textbox(label="RoBERTa Result")
                out_win = gr.Label(label="Comparison")
        
        btn_test.click(run_custom_test, inputs=[ctx_input, q_input], outputs=[out_fast, out_acc, out_win])

    with gr.Tab("2. Dataset Evaluation"):
        gr.Markdown("### Run SQuAD Benchmark")
        gr.Markdown("Click below to evaluate both models on a validation subset. (This runs live on your M2 chip!)")
        btn_eval = gr.Button("Run Benchmark Comparison", interactive=False)
        
        with gr.Row():
            plot_output = gr.Plot(label="Performance Chart")
            df_output = gr.Dataframe(label="Detailed Metrics")
            
        btn_eval.click(run_dataset_evaluation, outputs=[df_output, plot_output])

    with gr.Tab("3. Training (Excellent Rating)"):
        gr.Markdown("### 🚀 Model Fine-Tuning Experiments")
        gr.Markdown("""
        To achieve the **Excellent Rating**, I performed fine-tuning experiments to improve model performance.
        Since training requires GPU acceleration, I utilized **Google Colab**.
        """)
        
        # LINK TO COLAB
        gr.Markdown("""
        ### [🔗 Click here to view the Colab Training Notebook](https://colab.research.google.com/drive/1GQKJfb3xHWwQuVC4kSWwaUg2csvmmUf4?usp=sharing)
        *(Replace the link above with your actual shared Colab link)*
        """)
        
        gr.Info("Training logs and 'Before vs After' charts are available in the notebook linked above.")
    
    # Load models when page opens
    demo.load(
        fn=load_all_models,
        outputs=[loading_status, btn_test, btn_eval],
        show_progress="full"
    )

demo.launch()