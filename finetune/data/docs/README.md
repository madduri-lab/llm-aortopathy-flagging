# 🧬 Marfan LLM Data Curation

<p align="center" style="font-size: 18px;">
    <b>Marfan LLM Data Curation - Two Methods</b>
</p>

## 📚 Quick Start for data curation with GROBID
[What is GROBID](https://grobid.readthedocs.io/en/latest/Principles/)  
[Where is GROBID for data curation idea from](https://arxiv.org/pdf/2401.08406.pdf)
### GROBID Environment Setup
Docker is utilized for rapid deployment. 
If you need to run on an HPC platform, where docker cannot be used, you can use singularity or refer to the official [document](https://grobid.readthedocs.io/en/latest/Install-Grobid/) build from source.

```bash
docker pull grobid/grobid:${latest_grobid_version}
docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.0
```

### Run grobid.sh for text extraction
Before running grobid.sh, pls open the script and fill the config needed following instructions in the comment.
```bash
cd MarfanLLM/data/docs
./grobid.sh
```

## 📚 Quick Start for data curation with pymupdf
Run the script where file 'PMC_IDs.txt' is, this file should have the PMC IDs you want to download  
Set start_line and end_line to designate which part of the list you want to Run  
If 'SOURCE_DOCUMENTS' directory doesnt exist, the script will create one and output the PDFs there  
A logfile called 'download_logfile.txt' will be generated with status of download for each PDF as successful or failure  
The script should not stall if a download fails, it will continue on to next ID  
This script will print out status of each download incrementally to your terminal  