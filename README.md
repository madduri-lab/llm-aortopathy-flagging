# 🧬 Marfan LLM

<p align="center" style="font-size: 18px;">
    <b>Marfan Syndrome Specific LLM - Finetuned from LLaMA 2</b>
</p>

## Quick Start
### Environment Setup
```bash
conda create -n marfan-llm
conda activate marfan-llm
pip install -r requirements.txt
```

### MIMIC Data Download
- To utilize the [MIMIC dataset](https://physionet.org/content/mimic-iv-note/2.2/), please first make sure that you have access to the dataset by filling the applications and finishing the required training sessions. 
- Then run the following commands first to download the dataset into the `data/datasets/raw` folder. Make sure to replace `USERNAME` with your own PhysioNet username.
```bash
cd data/datasets/raw
wget -r -N -c -np --user USERNAME --ask-password https://physionet.org/files/mimic-iv-note/2.2/
cd physionet.org/files/mimic-iv-note/2.2/note && gunzip discharge.csv.gz
```

### MIMIC DATA Preprocess
To run the preprocess to the MIMIC dataset to extract notes for Marfan cases and controls, you can run the following `mimic_process.py` script in `data/datasets/raw` 
```bash
python mimic_process.py
```

### Marfan Syndrome Research Papers Download
- You can download all research papers related to Marfan Syndrome for retrieval from [Google Drive](https://drive.google.com/file/d/1vBWAB6tM0y0VUJ-cOQ3i9oMW9JEPFZdG/view?usp=share_link). 
- Then put all those PDF files into folder `data/docs`


### Mock Data Preparation
- Put the labeled training json files into folder `data/datasets/raw`
- Run `python ingest.py` first to setup local index DB.
- Run `python run_retriever.py` to get the retrival-augmented training json files into `data/datasets/rag`

## Ideas
Now the idea is adopted from the following paper from META AI.

<a href="https://arxiv.org/pdf/2310.01352.pdf">
    <img src="https://img.shields.io/badge/arXiv-2310.01352-B31B1B.svg" alt="radit">
</a>

