"""
Gradio UI layout and components.
"""
import gradio as gr
from samples import TEST_SAMPLES, get_sample_context, get_sample_questions, get_expected_answer
from evaluation import run_custom_test, run_dataset_evaluation, run_sample_comparison
from models import load_all_models


def create_ui():
    """Create and return the Gradio Blocks UI."""
    
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
                    ctx_input = gr.Textbox(
                        lines=5, 
                        label="Context", 
                        placeholder="Paste a paragraph here...", 
                        value="The MacBook Air is a line of laptop computers developed and manufactured by Apple Inc. It consists of a full-size keyboard, a machined aluminum case, and, in the more modern structure, a thin light structure."
                    )
                    q_input = gr.Textbox(
                        lines=2, 
                        label="Question", 
                        placeholder="Type a question...", 
                        value="Who manufactures the MacBook Air?"
                    )
                    btn_test = gr.Button("Ask Models", variant="primary", interactive=False)
                with gr.Column():
                    out_fast = gr.Textbox(label="DistilBERT Result")
                    out_acc = gr.Textbox(label="RoBERTa Result")
                    out_win = gr.Label(label="Comparison")
            
            btn_test.click(run_custom_test, inputs=[ctx_input, q_input], outputs=[out_fast, out_acc, out_win])

        with gr.Tab("2. Predefined Samples"):
            gr.Markdown("### 📋 Test with Predefined Samples")
            gr.Markdown("Select a sample and question to compare how both models perform. These samples are designed to highlight differences between DistilBERT and RoBERTa.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    sample_dropdown = gr.Dropdown(
                        choices=list(TEST_SAMPLES.keys()),
                        value=list(TEST_SAMPLES.keys())[0],
                        label="Select Sample Topic"
                    )
                    question_dropdown = gr.Dropdown(
                        choices=TEST_SAMPLES[list(TEST_SAMPLES.keys())[0]]["questions"],
                        value=TEST_SAMPLES[list(TEST_SAMPLES.keys())[0]]["questions"][0],
                        label="Select Question"
                    )
                    sample_context = gr.Textbox(
                        lines=6, 
                        label="Context (Read-only)", 
                        value=TEST_SAMPLES[list(TEST_SAMPLES.keys())[0]]["context"],
                        interactive=False
                    )
                    expected_answer = gr.Textbox(
                        label="✅ Expected Answer",
                        value=TEST_SAMPLES[list(TEST_SAMPLES.keys())[0]]["answers"][0],
                        interactive=False
                    )
                    btn_sample_test = gr.Button("🔍 Compare Models", variant="primary", interactive=False)
                
                with gr.Column(scale=1):
                    sample_out_fast = gr.Textbox(label="🚀 DistilBERT (Fast)")
                    sample_out_acc = gr.Textbox(label="🎯 RoBERTa (Accurate)")
                    sample_comparison = gr.Markdown(label="Analysis")
            
            # Update context and questions when sample changes
            sample_dropdown.change(
                fn=get_sample_context,
                inputs=[sample_dropdown],
                outputs=[sample_context]
            )
            sample_dropdown.change(
                fn=get_sample_questions,
                inputs=[sample_dropdown],
                outputs=[question_dropdown]
            )
            
            # Update expected answer when question changes
            def update_expected(sample_name, question):
                return get_expected_answer(sample_name, question)
            
            question_dropdown.change(
                fn=update_expected,
                inputs=[sample_dropdown, question_dropdown],
                outputs=[expected_answer]
            )
            sample_dropdown.change(
                fn=lambda s: get_expected_answer(s, TEST_SAMPLES[s]["questions"][0]),
                inputs=[sample_dropdown],
                outputs=[expected_answer]
            )
            
            btn_sample_test.click(
                fn=run_sample_comparison,
                inputs=[sample_dropdown, question_dropdown],
                outputs=[sample_out_fast, sample_out_acc, sample_comparison]
            )

        with gr.Tab("3. Dataset Evaluation"):
            gr.Markdown("### Run SQuAD Benchmark")
            gr.Markdown("Click below to evaluate both models on a validation subset. (This runs live on your M2 chip!)")
            btn_eval = gr.Button("Run Benchmark Comparison", interactive=False)
            
            with gr.Row():
                plot_output = gr.Plot(label="Performance Chart")
                df_output = gr.Dataframe(label="Detailed Metrics")
                
            btn_eval.click(run_dataset_evaluation, outputs=[df_output, plot_output])

        with gr.Tab("4. Training (Excellent Rating)"):
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
            outputs=[loading_status, btn_test, btn_sample_test, btn_eval],
            show_progress="full"
        )
    
    return demo
