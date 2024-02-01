# 🧬 Marfan LLM

<p align="center" style="font-size: 18px;">
    <b>Marfan Syndrome Specific LLM - Finetuned from LLaMA 2</b>
</p>

## 📚 Quick Start
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

## 🦙 Download LLaMa 2
1. To download LLaMa v2, first fill this [request form](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and obtain a download url via email.

2. Clone the llama repository.
    ```bash
    git clone https://github.com/facebookresearch/llama.git && cd llama
    ```

3. Run the download script, and you will need to provide the download url and choose the model size and type. The following instructions assume downloading 7B-chat version. **[Note: for this project, I am not sure which one is better, 7B or 7B-chat]**
    ```bash
    ./download.sh
    ```

4. Now we run the following two commands to convert the downloaded Llama model weights to HuggingFace weights to employ HuggingFace API for fine-tuning and evaluation later. [**Note**: When downloading Llama v2 7B-chat, the model will be downloaded into a folder named `llama-7b-chat`. We need to first rename the folder to `7B` by running `mv llama-7b-chat 7B`.]
    ```bash
    TRANSFORM=`python -c "import transformers;print('/'.join(transformers.__file__.split('/')[:-1])+'/models/llama/convert_llama_weights_to_hf.py')"`
    ```
    ```
    python ${TRANSFORM} --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
    ```


## Ideas
Now the idea is adopted from the following paper from META AI.

<a href="https://arxiv.org/pdf/2310.01352.pdf">
    <img src="https://img.shields.io/badge/arXiv-2310.01352-B31B1B.svg" alt="radit">
</a>

