## Multimodal-RAG

## How to run application?

## Creating Environment 
```bash
conda create -p venv python=3.10 -y

```
## Activating Environment
```bash
source activate venv/
```

## Installing dependencies
```bash
pip install -r requirements.txt
```
### To run the application
```bash
python app.py 
```


<!-- ### To run the application
```bash
uvicorn app:app
``` -->

## Install dependencies on Windows

1. Install poppler on windows and set the environment variable path.
- Unzip the downloaded folder.
- In C drive goto Program Files (x86) and create a folder name `poppler` and all the files from the uziped folder.
- `C:\Program Files (x86)\poppler\Library\bin` add this path to environment variables.

```
https://github.com/oschwartz10612/poppler-windows/releases/download/v24.07.0-0/Release-24.07.0-0.zip
```
2. Install Tessaract-ocr on windows and set the environment variable path.
```
https://github.com/UB-Mannheim/tesseract/releases/download/v5.4.0.20240606/tesseract-ocr-w64-setup-5.4.0.20240606.exe
```




### To know about Multimodal RAG
```bash
https://blog.langchain.dev/semi-structured-multi-modal-rag/
```
