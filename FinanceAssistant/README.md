#  Finance Domain-Specific LLM Assistant

> Fine-tuned **TinyLlama-1.1B** for financial Q&A using LoRA/PEFT — achieving **26% training loss reduction** and **28% ROUGE improvement** over the base model.

## link to deplyed Finance Domain-Specific LLM Assistant:https://huggingface.co/spaces/Caline0/finance-assistant



##  Project Overview

This project builds a **Finance Domain-Specific Assistant** by fine-tuning TinyLlama-1.1B-Chat on a curated dataset of financial instruction-response pairs. The assistant answers questions on:

-  Portfolio theory & asset allocation
- Equity valuation (P/E ratio, EPS, DCF)
- Fixed income & bond pricing
- Risk metrics (volatility, beta, VaR)
- Personal finance (compound interest, dollar-cost averaging)
- Market analysis & sentiment


## Example Conversations

**Q: What is portfolio diversification?**
> *Fine-tuned*: "Portfolio diversification spreads investments across multiple asset classes (stocks, bonds, real estate), sectors, and geographies to reduce risk. When one asset underperforms, others may compensate, lowering overall volatility. A classic 60/40 portfolio: 60% equities, 40% bonds..."

> *Base model*: "Diversification is spreading money around. It helps reduce risk in general investments..."

---

**Q: Explain the P/E ratio.**
> *Fine-tuned*: "The P/E ratio divides stock price by earnings per share (EPS). A P/E of 20 means investors pay $20 for every $1 of annual earnings. Low P/E (<15) may signal undervaluation; high P/E (>30) indicates growth optimism..."

> *Base model*: "P/E stands for price to earnings. It tells you how expensive a stock is..."


##  Gradio UI

The notebook launches an interactive **Gradio ChatInterface** at the end:
- Type any finance question in the input box
- Click one of the 7 example questions to auto-fill
- Chat history is maintained across turns
- Public share link generated automatically 

## Repository Structure


├── fine_tune_finance_assistant.ipynb   
├── README.md                           
├── requirements.txt                    
├── inference.py                        
└── hf_space/
    ├── app.py                          
    ├── requirements.txt
    └── README.md


##  Dependencies

See `requirements.txt`. Key packages:
- `transformers>=4.36.0`
- `peft>=0.4.0`
- `trl>=0.7.0`
- `bitsandbytes>=0.41.0`
- `gradio>=4.0.0`
- `datasets>=2.14.0`
- `rouge-score`, `nltk`
