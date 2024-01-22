# 🧬 Marfan LLM

<p align="center" style="font-size: 18px;">
    <b>Marfan Syndrome Specific LLM - Finetuned from LLaMA 2</b>
</p>

## Quick Start
### Environment Setup
```bash
conda create -n marfan-llm
conda activate marfan-llm
pip install -r requirement.txt
```
### Data Preparation
- Put source pdf documents into folder `data/docs`
- Put the labeled training json files into folder `data/datasets/raw`
- Run `python ingest.py` first to setup local index DB.
- Run `python run_retriever.py` to get the retrival-augmented training json files into `data/datasets/rag`

## Ideas
Now the idea is adopted from the following paper from META AI.

<a href="https://arxiv.org/pdf/2310.01352.pdf">
    <img src="https://img.shields.io/badge/arXiv-2310.01352-B31B1B.svg" alt="radit">
</a>

