"""
Evaluation and comparison functions.
"""
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import config
from samples import TEST_SAMPLES, get_expected_answer


def run_custom_test(context, question):
    """Runs inference on a single custom sample."""
    if not config.models_ready:
        return "⏳ Models still loading...", "⏳ Please wait...", "Loading..."
    
    res1 = config.qa_fast(question=question, context=context)
    res2 = config.qa_accurate(question=question, context=context)
    
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
    if not config.models_ready:
        return pd.DataFrame(), None
    
    results_list = []
    models = [("DistilBERT", config.qa_fast), ("RoBERTa", config.qa_accurate)]
    
    progress(0, desc="Starting Evaluation...")
    total_steps = len(models) * len(config.squad_subset)
    current_step = 0
    
    for model_name, model_pipe in models:
        predictions = []
        references = []
        pred_texts = []
        ref_texts = []
        
        # Loop through dataset with progress update
        for i, item in enumerate(config.squad_subset):
            current_step += 1
            progress(current_step / total_steps, desc=f"Testing {model_name}...")
            
            res = model_pipe(question=item['question'], context=item['context'])
            
            predictions.append({
                'id': item['id'], 
                'prediction_text': res['answer'],
                'no_answer_probability': 0.0
            })
            references.append({'id': item['id'], 'answers': item['answers']})
            
            # Store text for BERTScore and ROUGE
            pred_texts.append(res['answer'])
            ref_text = item['answers']['text'][0] if item['answers']['text'] else ""
            ref_texts.append(ref_text)
            
        # Compute SQuAD metrics
        squad_score = config.metric.compute(predictions=predictions, references=references)
        
        # Compute BERTScore (filter out empty references)
        valid_pairs = [(p, r) for p, r in zip(pred_texts, ref_texts) if r]
        if valid_pairs:
            valid_preds, valid_refs = zip(*valid_pairs)
            bert_scores = config.bertscore_metric.compute(
                predictions=list(valid_preds), 
                references=list(valid_refs), 
                lang="en"
            )
            avg_bertscore = sum(bert_scores['f1']) / len(bert_scores['f1'])
        else:
            avg_bertscore = 0.0
        
        # Compute ROUGE
        if valid_pairs:
            rouge_scores = config.rouge_metric.compute(
                predictions=list(valid_preds), 
                references=list(valid_refs)
            )
            rouge_l = rouge_scores['rougeL']
        else:
            rouge_l = 0.0
        
        results_list.append({
            "Model": model_name, 
            "F1": round(squad_score['f1'], 4), 
            "Exact Match": round(squad_score['exact'], 4),
            "BERTScore": round(avg_bertscore, 4),
            "ROUGE-L": round(rouge_l, 4)
        })
    
    # Create Dataframe
    df = pd.DataFrame(results_list)
    
    # Generate Plot with all metrics
    fig, ax = plt.subplots(figsize=(8, 5))
    df.set_index('Model')[['F1', 'Exact Match', 'BERTScore', 'ROUGE-L']].plot(
        kind='bar', ax=ax, 
        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    )
    ax.set_title("Model Performance on SQuAD (Subset)")
    ax.set_ylabel("Score")
    ax.set_xticklabels(df['Model'], rotation=0)
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    return df, fig


def run_sample_comparison(sample_name, question):
    """Compare models on a predefined sample."""
    if not config.models_ready:
        return "⏳ Models loading...", "⏳ Please wait...", "Loading..."
    
    context = TEST_SAMPLES[sample_name]["context"]
    expected = get_expected_answer(sample_name, question)
    
    res1 = config.qa_fast(question=question, context=context)
    res2 = config.qa_accurate(question=question, context=context)
    
    out1 = f"**Answer:** {res1['answer']}\n**Confidence:** {res1['score']:.4f}"
    out2 = f"**Answer:** {res2['answer']}\n**Confidence:** {res2['score']:.4f}"
    
    # Detailed comparison
    analysis = f"""
### 📊 Comparison Analysis

| Metric | DistilBERT | RoBERTa |
|--------|------------|---------|
| **Answer** | {res1['answer']} | {res2['answer']} |
| **Confidence** | {res1['score']:.4f} | {res2['score']:.4f} |
| **Expected** | {expected} | {expected} |

"""
    # Determine correctness - normalize and compare
    def normalize_answer(s):
        """Normalize answer for comparison."""
        import re
        s = s.lower().strip()
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = ' '.join(s.split())
        return s
    
    def is_correct(prediction, expected_ans):
        """Check if prediction matches expected answer."""
        pred_norm = normalize_answer(prediction)
        exp_norm = normalize_answer(expected_ans)
        
        # Exact match after normalization
        if pred_norm == exp_norm:
            return True
        
        # For partial matches, ensure negative signs are preserved
        if exp_norm in pred_norm or pred_norm in exp_norm:
            # If expected has a negative sign, prediction must too
            if '-' in expected_ans and '-' not in prediction:
                return False
            return True
        
        return False
    
    dist_correct = is_correct(res1['answer'], expected)
    rob_correct = is_correct(res2['answer'], expected)
    
    if "[Unanswerable" in expected:
        # For unanswerable questions, lower confidence is better
        if res2['score'] < res1['score']:
            analysis += "**🏆 Winner: RoBERTa** - Shows lower confidence on unanswerable question (trained on SQuAD2 with unanswerable examples)"
        else:
            analysis += "**🏆 Winner: DistilBERT** - Shows lower confidence on this unanswerable question"
    elif dist_correct and rob_correct:
        if res2['score'] > res1['score']:
            analysis += "**🏆 Winner: RoBERTa** - Both correct, but RoBERTa is more confident"
        elif res1['score'] > res2['score']:
            analysis += "**🏆 Winner: DistilBERT** - Both correct, and DistilBERT is more confident (faster too!)"
        else:
            analysis += "**🤝 Tie** - Both models got the correct answer with similar confidence"
    elif rob_correct and not dist_correct:
        analysis += "**🏆 Winner: RoBERTa** - Only RoBERTa found the correct answer!"
    elif dist_correct and not rob_correct:
        analysis += "**🏆 Winner: DistilBERT** - Only DistilBERT found the correct answer!"
    else:
        analysis += "**❌ Neither model found the expected answer** - This is a challenging question"
    
    return out1, out2, analysis
