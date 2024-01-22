Idea comes from https://arxiv.org/pdf/2310.01352.pdf  

How to run:
1. Run pip install -r requirement.txt to setup environment.  
2. Put source documents into the folder "SOURCE_DOCUMENTS".  
3. Run ingest.py first to setup local DB.  
4. Put source json files into the folder DATA/RAW.  
5. Run run_retriever.py to get RAG jsons. All files will be stored into DATA/RAG.