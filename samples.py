"""
Predefined test samples and helper functions.
"""
import gradio as gr

# --- Predefined Test Samples ---
TEST_SAMPLES = {
    "1. AI & Machine Learning (Technical)": {
        "context": "Transformers are a type of neural network architecture introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017. Unlike recurrent neural networks (RNNs), transformers process all input tokens simultaneously using self-attention mechanisms. BERT, developed by Google in 2018, is a bidirectional transformer pre-trained on masked language modeling. RoBERTa, created by Facebook AI in 2019, improved upon BERT by training longer, on more data, and removing the next sentence prediction objective.",
        "questions": [
            "Who developed RoBERTa?",
            "What did RoBERTa remove from BERT's training?",
            "When were transformers introduced?"
        ],
        "answers": ["Facebook AI", "next sentence prediction objective", "2017"]
    },
    "2. History (Manhattan Project)": {
        "context": "The Manhattan Project was a secret research program that produced the first nuclear weapons during World War II. Led by physicist J. Robert Oppenheimer, the project employed over 125,000 workers and cost nearly $2 billion (equivalent to about $28 billion today). The first nuclear device was tested on July 16, 1945, at the Trinity site in New Mexico. Less than a month later, atomic bombs were dropped on Hiroshima on August 6 and Nagasaki on August 9, 1945.",
        "questions": [
            "Who led the Manhattan Project?",
            "How many workers were employed?",
            "Where was the first nuclear device tested?"
        ],
        "answers": ["J. Robert Oppenheimer", "over 125,000", "Trinity site in New Mexico"]
    },
    "3. Business (Tesla - Tricky)": {
        "context": "Tesla, Inc. is an American electric vehicle and clean energy company founded in 2003 by Martin Eberhard and Marc Tarpenning. Elon Musk joined as chairman in 2004 after leading the Series A funding round, and became CEO in 2008. The company's first vehicle, the Roadster, was launched in 2008. Tesla's Model S sedan, introduced in 2012, became the best-selling plug-in electric car worldwide in 2015 and 2016. The more affordable Model 3 was released in 2017.",
        "questions": [
            "Who founded Tesla?",
            "When did Elon Musk become CEO?",
            "Which Tesla model became the best-selling plug-in electric car?"
        ],
        "answers": ["Martin Eberhard and Marc Tarpenning", "2008", "Model S"]
    },
    "4. Sports (2022 World Cup)": {
        "context": "The 2022 FIFA World Cup was held in Qatar from November 20 to December 18, 2022. It was the first World Cup held in the Arab world and the second held entirely in Asia after the 2002 tournament in South Korea and Japan. Argentina won the tournament, defeating France 4-2 on penalties after a 3-3 draw in the final. Lionel Messi was named the Best Player of the tournament, while Kylian Mbappé won the Golden Boot with 8 goals.",
        "questions": [
            "Which country won the 2022 World Cup?",
            "Who won the Golden Boot?",
            "How many goals did Mbappé score?"
        ],
        "answers": ["Argentina", "Kylian Mbappé", "8"]
    },
    "5. Legal (GDPR - Unanswerable Test)": {
        "context": "The European Union's General Data Protection Regulation (GDPR) came into effect on May 25, 2018. It replaced the 1995 Data Protection Directive and strengthened data protection rules for individuals within the EU. The regulation applies to all companies processing personal data of EU residents, regardless of the company's location. Fines for violations can reach up to €20 million or 4% of annual global revenue, whichever is higher.",
        "questions": [
            "When did GDPR come into effect?",
            "What is the maximum fine percentage of revenue?",
            "Who is the current GDPR commissioner?"
        ],
        "answers": ["May 25, 2018", "4%", "[Unanswerable - not in text]"]
    },
    "6. Science (Negation & Nuance Test)": {
        "context": "Water freezes at 0 degrees Celsius at standard atmospheric pressure. However, pure water can sometimes remain liquid below 0°C in a phenomenon called supercooling. Salt water does not freeze at 0°C; it freezes at approximately -2°C depending on salinity. Mercury, unlike water, freezes at -39°C. Interestingly, helium is the only element that does not freeze at normal atmospheric pressure regardless of temperature.",
        "questions": [
            "At what temperature does salt water freeze?",
            "What element does not freeze at normal pressure?",
            "At what temperature does mercury freeze?"
        ],
        "answers": ["approximately -2°C", "helium", "-39°C"]
    }
}


def get_sample_context(sample_name):
    """Return context for selected sample."""
    if sample_name in TEST_SAMPLES:
        return TEST_SAMPLES[sample_name]["context"]
    return ""


def get_sample_questions(sample_name):
    """Return questions dropdown choices for selected sample."""
    if sample_name in TEST_SAMPLES:
        return gr.Dropdown(choices=TEST_SAMPLES[sample_name]["questions"], value=TEST_SAMPLES[sample_name]["questions"][0])
    return gr.Dropdown(choices=[], value=None)


def get_expected_answer(sample_name, question):
    """Return expected answer for display."""
    if sample_name in TEST_SAMPLES:
        sample = TEST_SAMPLES[sample_name]
        if question in sample["questions"]:
            idx = sample["questions"].index(question)
            return sample["answers"][idx]
    return ""
