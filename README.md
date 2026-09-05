# N8N RAG Agent with OCR

A complete Retrieval-Augmented Generation (RAG) pipeline with OCR capabilities, built with N8N workflows and FastAPI OCR services.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Google Drive │────▶│  File Router  │────▶│  OCR/API    │
│   Trigger   │     │  Workflow     │     │  Services   │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Pinecone    │
                    │  Vector DB   │
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Knowledge    │
                    │  Assistant   │
                    └──────────────┘
```

## Components

### OCR Services (FastAPI)

| Service | Endpoint | Description |
|---------|----------|-------------|
| `pdf_ocr_hybrid.py` | `/pdf_ocr` | Hybrid PaddleOCR + EasyOCR with preprocessing |
| `pdf_ocr_fast_api.py` | `/pdf_fast_ocr` | Adaptive PaddleOCR (mobile/server models) |
| `pdf_ocr_api.py` | `/pdf_ocr` | Simple EasyOCR + PyMuPDF |
| `ocr_api.py` | `/ocr` | Single image OCR |

### N8N Workflows

| Workflow | Purpose |
|----------|---------|
| `Knowledge_Assistant.json` | RAG chat agent with Pinecone, Ollama, Cohere |
| `A_File_Router_Workflow.json` | Google Drive → file routing → OCR/extract → Pinecone |

## Quick Start

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start OCR API Server

```bash
# Hybrid OCR (recommended)
uvicorn pdf_ocr_hybrid:app --host 0.0.0.0 --port 8000

# Or fast adaptive OCR
uvicorn pdf_ocr_fast_api:app --host 0.0.0.0 --port 8000
```

### 3. Configure N8N Workflows

Import the workflow templates and configure credentials:

```bash
# Copy templates and fill in your credentials
cp workflows/Knowledge_Assistant.template.json workflows/Knowledge_Assistant.json
cp workflows/A_File_Router_Workflow.template.json workflows/A_File_Router_Workflow.json
```

Required credentials in N8N:
- **Pinecone API** - Vector database
- **Ollama API** - Embeddings (nomic-embed-text) + LLM (llama3.1:8b)
- **Cohere API** - Reranker
- **Google Drive OAuth2** - File trigger

### 4. Set Environment Variables

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

### Required Services

- **Pinecone** - Index named `rag` (dimension 768 for nomic-embed-text)
- **Ollama** - Running locally with models:
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3.1:8b
  ```
- **Cohere** - API key for reranking
- **Google Drive** - OAuth2 credentials for file watch

### OCR API Endpoints

The workflows expect OCR services at:
- `http://127.0.0.1:8000/ocr` - Image OCR
- `http://127.0.0.1:8000/pdf_fast_ocr` - PDF OCR (fast)
- `http://127.0.0.1:8000/pdf_ocr` - PDF OCR (hybrid)

## Project Structure

```
N8N_RAG_Agent/
├── .gitignore
├── requirements.txt
├── README.md
├── .env.example
├── pdf_ocr_hybrid.py        # Hybrid PaddleOCR + EasyOCR
├── pdf_ocr_fast_api.py      # Adaptive PaddleOCR
├── pdf_ocr_api.py           # Simple EasyOCR
├── ocr_api.py               # Image OCR endpoint
├── ocr.py                   # PaddleOCR test
├── paddle_test.py           # Direct PDF test
├── workflows/
│   ├── Knowledge_Assistant.template.json
│   └── A_File_Router_Workflow.template.json
└── ocr_env/                 # Virtual env (gitignored)
```

## Workflow Details

### File Router Workflow

1. **Google Drive Trigger** - Watches folder for new files
2. **Switch Node** - Routes by file extension:
   - Images (jpg, png, etc.) → OCR API
   - PDFs → Extract text → Check if scanned → OCR if needed
   - Text files (txt, csv, json, md, docx, code) → Direct extract
   - Unknown → Mark for review
3. **Merge & Normalize** - Combine metadata, clean text
4. **Pinecone Insert** - Chunk (200 overlap) → Embed → Store

### Knowledge Assistant

1. **Chat Trigger** - Receives user messages
2. **AI Agent** - Uses Pinecone as tool with reranker
3. **Memory** - Window buffer for conversation context
4. **Embeddings** - Ollama nomic-embed-text
5. **LLM** - Ollama llama3.1:8b
6. **Reranker** - Cohere for relevance scoring

## License

MIT