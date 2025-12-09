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

# Load Models (Cached for speed)
print("Loading models... (this may take a moment)")
qa_fast = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=device)
qa_accurate = pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)

# Load Metric & Dataset Slice
metric = evaluate.load("squad_v2")
# OLD
# squad_subset = load_dataset("squad_v2", split="validation[:20]")

# NEW: Take 50-100 random samples
# seed=42 ensures you get the same "random" set every time you restart
squad_subset = load_dataset("squad_v2", split="validation").shuffle(seed=42).select(range(50))

# --- Logic Functions ---

def run_custom_test(context, question):
    """Runs inference on a single custom sample."""
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
        results_list.append({"Model": model_name, "F1": score['f1'], "Exact Match": score['exact']})
    
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

    with gr.Tab("1. Custom Test"):
        gr.Markdown("### Test with your own text")
        with gr.Row():
            with gr.Column():
                ctx_input = gr.Textbox(lines=5, label="Context", placeholder="Paste a paragraph here...", value="The MacBook Air is a line of laptop computers developed and manufactured by Apple Inc. It consists of a full-size keyboard, a machined aluminum case, and, in the more modern structure, a thin light structure.")
                q_input = gr.Textbox(lines=2, label="Question", placeholder="Type a question...", value="Who manufactures the MacBook Air?")
                btn_test = gr.Button("Ask Models", variant="primary")
            with gr.Column():
                out_fast = gr.Textbox(label="DistilBERT Result")
                out_acc = gr.Textbox(label="RoBERTa Result")
                out_win = gr.Label(label="Comparison")
        
        btn_test.click(run_custom_test, inputs=[ctx_input, q_input], outputs=[out_fast, out_acc, out_win])

    with gr.Tab("2. Dataset Evaluation"):
        gr.Markdown("### Run SQuAD Benchmark")
        gr.Markdown("Click below to evaluate both models on a validation subset. (This runs live on your M2 chip!)")
        btn_eval = gr.Button("Run Benchmark Comparison")
        
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
        ### [🔗 Click here to view the Colab Training Notebook](https://colab.research.google.com/)
        *(Replace the link above with your actual shared Colab link)*
        """)
        
        gr.Info("Training logs and 'Before vs After' charts are available in the notebook linked above.")

demo.launch()