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

### Marfan Syndrome Research Papers Download
- You can download all research papers related to Marfan Syndrome for retrieval from [Google Drive](https://drive.google.com/file/d/1vBWAB6tM0y0VUJ-cOQ3i9oMW9JEPFZdG/view?usp=share_link). 
- Then put all those PDF files into folder `data/docs`

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


## Retrieve Contexts for the Clinical Note Summaries
To retrieve relevant contexts from marfan syndrome research papers for the marfan patient clinical notes at `data/datasets/raw/marfan_allnotes_022024.csv`, we need to do the following
1. Make sure you have already downloaded the research papers and put them inside the `data/docs` folder.
2. Create the vector database by running the following script.
```bash
python ingest.py
```
3. Retrieve contexts from the vector database for the patient clinical notes by running the following script. This will generate two JSON files `data/datasets/rag/marfan_rag_train.json` and `data/datasets/rag/marfan_rag_test.json`.
```bash
python preprocess.py
```

## Supervised fine-tuning on Marfan Notes
```bash
python finetune.py --output_name ./model/marfan_prediction/lora_7B.pt --dataset_type ClinicalNoteDataset --train_data_path ./data/datasets/rag/marfan_rag_train.json --validation_data_path ./data/datasets/rag/marfan_rag_test.json --batch_size_training 1 --batch_size_validation 1 --lr 1e-4 --weight_decay 0.01 --gamma 0.95 --num_epochs 1 --gradient_accumulation_steps 2
```

## 📖 Unsupervised fine-tuning on raw text
```bash
python finetune.py --output_name ./model/marfan/lora_7B.pt --dataset_type RawTextDataset --train_data_path ./data/datasets/raw/sample_train.txt --validation_data_path ./data/datasets/raw/sample_test.txt --max_tokens 4096 --is_ntp --batch_size_training 4 --batch_size_validation 1 --lr 2e-5 --weight_decay 0.01 --gamma 0.95 --num_epochs 1
```