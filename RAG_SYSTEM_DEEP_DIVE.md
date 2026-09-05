# N8N RAG Agent with OCR - Complete System Deep Dive

> **A comprehensive guide to understanding Retrieval-Augmented Generation (RAG) and this production-ready implementation**

---

## Table of Contents

1. [What is RAG? (Beginner Friendly)](#what-is-rag)
2. [Why RAG? Problems It Solves](#why-rag)
3. [System Architecture Overview](#architecture)
4. [Core Components Deep Dive](#components)
5. [Data Flow Walkthrough](#data-flow)
6. [OCR Pipeline Explained](#ocr-pipeline)
7. [Vector Databases & Embeddings](#vector-db)
8. [The Retrieval Process](#retrieval)
9. [Generation & Answer Synthesis](#generation)
10. [N8N Workflow Architecture](#n8n-workflow)
11. [Production Considerations](#production)
12. [How to Extend & Customize](#extend)
13. [Troubleshooting Guide](#troubleshooting)

---

## 1. What is RAG? (Beginner Friendly) {#what-is-rag}

### The Simple Analogy

Imagine you're a **librarian** who has read every book in the library. Someone asks you a question. You have two choices:

| Approach | How It Works | Problem |
|----------|--------------|---------|
| **Pure LLM (No RAG)** | Answer from memory | Hallucinates, outdated, no sources |
| **RAG** | 1. Search library catalog<br>2. Pull relevant books<br>3. Read relevant pages<br>4. Answer using ONLY those pages | Accurate, cited, up-to-date |

### Technical Definition

**RAG = Retrieval + Augmented + Generation**

```
User Question
     │
     ▼
┌─────────────────────────────────────┐
│  RETRIEVAL                          │
│  • Convert question to vector       │
│  • Search vector database           │
│  • Find top-K relevant chunks       │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  AUGMENTATION                       │
│  • Combine question + chunks        │
│  • Build context-rich prompt        │
│  • Include citations/metadata       │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  GENERATION                         │
│  • LLM generates answer             │
│  • Grounded in retrieved context    │
│  • Cites sources                    │
└─────────────────────────────────────┘
```

### Key Insight

> **The LLM doesn't "know" your documents. It only "reads" what you give it at query time.**

---

## 2. Why RAG? Problems It Solves {#why-rag}

### Problems with Pure LLMs

| Problem | Example | RAG Solution |
|---------|---------|--------------|
| **Hallucination** | "Our CEO is Elon Musk" (false) | Only answers from retrieved docs |
| **Knowledge Cutoff** | Doesn't know 2024 events | Documents uploaded anytime |
| **No Citations** | "Studies show..." (which?) | Returns `filename, page X` |
| **Private Data** | Can't access internal docs | Your docs → your vector DB |
| **Context Limit** | 4K-128K tokens max | Retrieves only relevant chunks |

### When to Use RAG

- ✅ Company knowledge base
- ✅ Legal/medical document search
- ✅ Customer support from manuals
- ✅ Research paper analysis
- ✅ Code documentation Q&A
- ❌ General knowledge (use pure LLM)
- ❌ Creative writing (use pure LLM)

---

## 3. System Architecture Overview {#architecture}

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE SYSTEM ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  DATA SOURCES │────▶│  INGESTION   │────▶│  PROCESSING  │────▶│  VECTOR DB   │
│              │     │  PIPELINE    │     │  & CHUNKING  │     │  (Pinecone)  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │                    │
       │                    │                    │                    │
       ▼                    ▼                    ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Google Drive │     │ File Router  │     │ Text Splitter│     │ Index: "rag" │
│   Folder     │     │  Workflow    │     │ (200 overlap)│     │ Dim: 768     │
│              │     │              │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                   ┌──────────────┐
                   │ OCR SERVICES │
                   │ (FastAPI)    │
                   └──────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
       ┌──────────┐  ┌──────────┐  ┌──────────┐
       │  Images  │  │  Scanned │  │ Digital  │
       │  (jpg/   │  │   PDFs   │  │   PDFs   │
       │   png)   │  │          │  │          │
       └──────────┘  └──────────┘  └──────────┘
              │             │             │
              ▼             ▼             ▼
       ┌──────────────────────────────────────┐
       │        Pinecone Vector Store         │
       │  • Embeddings: nomic-embed-text      │
       │  • Chunks with metadata              │
       │  • Reranker: Cohere                  │
       └──────────────────────────────────────┘
                            │
                            ▼
                   ┌──────────────┐
                   │  QUERY TIME  │
                   │  (Chat)      │
                   └──────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
       ┌──────────┐  ┌──────────┐  ┌──────────┐
       │ Embed    │  │ Retrieve │  │ Generate │
       │ Question │  │ Top-K    │  │  Answer  │
       │          │  │ + Rerank │  │ + Cite   │
       └──────────┘  └──────────┘  └──────────┘
              │             │             │
              ▼             ▼             ▼
       ┌──────────────────────────────────────┐
       │        Knowledge Assistant           │
       │  • LLM: llama3.1:8b (Ollama)         │
       │  • Memory: Window buffer             │
       │  • Tool: Pinecone Vector Store       │
       └──────────────────────────────────────┘
```

---

## 4. Core Components Deep Dive {#components}

### 4.1 Document Ingestion Pipeline (`A_File_Router_Workflow.json`)

This is the **ETL (Extract, Transform, Load)** pipeline that runs automatically when files are added to Google Drive.

#### Step-by-Step Flow:

```
Google Drive Trigger (every minute)
         │
         ▼
    Download File
         │
         ▼
    ┌─────────────────────────────────────┐
    │        SWITCH NODE (File Router)    │
    │  Routes by file extension           │
    └─────────────────────────────────────┘
         │
    ┌────┴────┬────────────────┬────────────┐
    ▼         ▼                ▼            ▼
 IMAGES    PDFs             TEXT FILES   UNKNOWN
 (jpg,     (pdf)            (txt, csv,   (zip, exe,
  png,     ─────────        json, md,    apk, etc.)
  webp)         │                   docx,      │
                │                   py, js)    │
                ▼                   │           ▼
         ┌───────────────┐          │    ┌────────────┐
         │  OCR API      │          ▼    │  Mark as   │
         │  (EasyOCR)    │    ┌───────────┐  │ unknown    │
         └───────────────┘    │ Extract   │  └────────────┘
                              │ from File │
                              │ (PyMuPDF) │
                              └───────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │ PDF Text Check      │
                          │ (length < 100?)     │
                          └─────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
             SCANNED PDF                  DIGITAL PDF
                    │                         │
                    ▼                         ▼
            ┌─────────────┐           ┌─────────────┐
            │ OCR API     │           │ Metadata    │
            │ (PaddleOCR  │           │ Builder     │
            │  adaptive)  │           │ (digital)   │
            └─────────────┘           └─────────────┘
                    │                         │
                    └────────────┬────────────┘
                                 ▼
                        ┌─────────────────┐
                        │ Merge & Normalize│
                        │ Metadata        │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Text Cleaner    │
                        │ • Normalize WS  │
                        │ • Remove repeats│
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Pinecone Insert │
                        │ • Chunk (200)   │
                        │ • Embed         │
                        │ • Upsert        │
                        └─────────────────┘
```

#### File Type Handling Matrix

| Extension | Route | Processing |
|-----------|-------|------------|
| `jpg, png, jpeg, webp, bmp, gif` | **IMAGES** | Direct OCR → EasyOCR |
| `pdf` | **PDFs** | Extract → Check if scanned → OCR if needed |
| `txt, csv, json, xml, md, docx, doc, c, cpp, py, js, xlsx, java` | **TEXT_FILES** | Direct text extraction |
| `zip, rar, exe, apk` | **UNKNOWN** | Mark for manual review |

### 4.2 OCR Services (FastAPI)

Three OCR implementations with different tradeoffs:

#### A. `pdf_ocr_hybrid.py` - **Recommended for Production**

```python
# Hybrid approach: Best of both worlds
1. PyMuPDF extracts native text first
2. If page has >100 chars → digital (skip OCR)
3. If page has <100 chars → scanned → OCR
4. PaddleOCR primary, EasyOCR fallback
5. Image preprocessing (grayscale, blur, threshold)
6. Confidence scoring per page
7. Engine selection per page
```

**Returns:**
```json
{
  "text": "full extracted text",
  "pages": 10,
  "digital_pages": 3,
  "scanned_pages": 7,
  "document_type": "hybrid_pdf",
  "confidence": 0.92,
  "warning_flag": false,
  "engines_used": ["paddle", "easyocr"]
}
```

#### B. `pdf_ocr_fast_api.py` - **Speed Optimized**

```python
# Adaptive multi-pass OCR
1. Try mobile model at 120 DPI (fast)
2. If confidence < 0.88 → retry at 150 DPI
3. If confidence < 0.83 → try server model at 180 DPI
4. Page-level timing for monitoring
5. Image resize to max 1400px width
```

**Best for:** High-volume, latency-sensitive applications

#### C. `pdf_ocr_api.py` - **Simple Baseline**

```python
# Simple pipeline
1. PyMuPDF text extraction
2. If insufficient → 400 DPI render
3. OTSU thresholding
4. EasyOCR (English + Hindi)
```

#### D. `ocr_api.py` - **Single Image OCR**

```python
# For individual images
POST /ocr
- Input: image file
- Output: text, confidence, blocks
```

### 4.3 Knowledge Assistant (`Knowledge_Assistant.template.json`)

The **query-time** RAG pipeline:

```
User Message
     │
     ▼
┌────────────────────────────────────┐
│        AI AGENT (LangChain)        │
│  • System prompt with instructions │
│  • Tools: Pinecone Vector Store    │
│  • Max iterations: 3               │
└────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────┐
│   PINECONE VECTOR STORE (Tool)     │
│  • Mode: retrieve-as-tool          │
│  • Index: "rag"                    │
│  • Embeddings: nomic-embed-text    │
│  • Reranker: Cohere                │
│  • Top-K retrieval + rerank        │
└────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────┐
│      NORMALIZE METADATA (Code)     │
│  • Flatten metadata structure      │
│  • Ensure consistent fields        │
│  • filename, document_type, pages  │
│  • confidence, pageContent         │
└────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────┐
│         LLM GENERATION             │
│  • Model: llama3.1:8b (Ollama)     │
│  • numPredict: 512                 │
│  • Context: retrieved chunks       │
│  • Memory: conversation history    │
└────────────────────────────────────┘
```

---

## 5. Data Flow Walkthrough {#data-flow}

### 5.1 Ingestion Flow (Document → Vector DB)

```
USER UPLOADS FILE TO GOOGLE DRIVE
                │
                ▼
┌────────────────────────────────────────────────────────────┐
│ 1. GOOGLE DRIVE TRIGGER                                    │
│    • Polls every minute                                    │
│    • Watches specific folder ID                            │
│    • Triggers on fileCreated event                         │
└────────────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────┐
│ 2. DOWNLOAD FILE                                           │
│    • Uses Google Drive OAuth2 credentials                  │
│    • Gets binary data + metadata (filename, extension)     │
└────────────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────┐
│ 3. SWITCH NODE - FILE TYPE ROUTING                         │
│    • 20+ extension conditions                              │
│    • Outputs: IMAGE_FILES, TEXT_FILES, UNKNOWN_FILES       │
└────────────────────────────────────────────────────────────┘
                │
       ┌────────┼────────┐
       ▼        ▼        ▼
   IMAGES    PDFs     TEXT_FILES
       │        │           │
       ▼        ▼           ▼
   OCR API  Extract    Extract
   (Easy)   Text      Text
       │     (PyMuPDF)   (PyMuPDF)
       │        │           │
       │        ▼           │
       │  PDF Text Check    │
       │  (length > 50?)    │
       │        │           │
       │    ┌───┴───┐       │
       │    ▼       ▼       │
       │ SCANNED  DIGITAL   │
       │    │        │      │
       │    ▼        ▼      │
       │ OCR API  Metadata  │
       │(Paddle)  Builder   │
       │    │        │      │
       └────┼────────┼──────┘
            │        │
            ▼        ▼
       ┌─────────────────────┐
       │ MERGE (Preserve     │
       │  Metadata)          │
       └─────────────────────┘
                │
                ▼
       ┌─────────────────────┐
       │ CODE: Normalize     │
       │ filename, extension │
       └─────────────────────┘
                │
                ▼
       ┌─────────────────────┐
       │ TEXT CLEANER        │
       │ • \s+ → single space│
       │ • Remove repeated   │
       │   chars (aaaaa→a)   │
       │ • Trim              │
       └─────────────────────┘
                │
                ▼
       ┌─────────────────────┐
       │ PINECONE VECTOR     │
       │ STORE (Insert)      │
       │ • Data Loader       │
       │ • Text Splitter     │
       │   (Recursive,       │
       │    chunk=200,       │
       │    overlap=200)     │
       │ • Embeddings        │
       │   (Ollama)          │
       │ • Upsert to index   │
       └─────────────────────┘
```

### 5.2 Query Flow (Question → Answer)

```
USER ASKS QUESTION IN CHAT
                │
                ▼
┌────────────────────────────────────────────────────────────┐
│ 1. CHAT TRIGGER                                            │
│    • Webhook receives message                              │
│    • Passes to AI Agent                                    │
└────────────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────┐
│ 2. AI AGENT DECIDES TO USE TOOL                            │
│    • System prompt: "Always use Pinecone_Vector_Store"     │
│    • Tool description enforces retrieval                   │
│    • Calls Pinecone Vector Store                           │
└────────────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────┐
│ 3. PINECONE VECTOR STORE (Retrieve)                        │
│    • Embed question (nomic-embed-text via Ollama)          │
│    • Vector similarity search (cosine)                     │
│    • Get top-K chunks                                      │
│    • Rerank with Cohere                                    │
│    • Return: pageContent + metadata                        │
└────────────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────┐
│ 4. NORMALIZE METADATA (Code Node)                          │
│    • Flattens nested metadata                              │
│    • Ensures: filename, document_type, pages, confidence  │
│    • Passes clean chunks to LLM                            │
└────────────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────┐
│ 5. LLM GENERATION (Ollama llama3.1:8b)                     │
│    • Context: Retrieved chunks + conversation memory       │
│    • System prompt enforces:                               │
│      - Never answer from memory                            │
│      - Cite sources [filename, page X]                     │
│      - Say "I don't know" if no results                    │
│    • Max 512 tokens                                        │
└────────────────────────────────────────────────────────────┘
                │
                ▼
         ANSWER RETURNED TO USER
```

---

## 6. OCR Pipeline Explained {#ocr-pipeline}

### Why OCR is Critical for RAG

> **Without OCR, scanned PDFs and images are invisible to RAG.** They become "dark data" - stored but unsearchable.

### OCR Decision Tree

```
PDF Page
    │
    ▼
┌─────────────────┐
│ Native Text     │
│ Extraction      │
│ (PyMuPDF)       │
└─────────────────┘
    │
    ▼
Text length > 100 chars?
    │
    ├─ YES ──▶ DIGITAL PAGE ──▶ Use native text (confidence=1.0)
    │
    └─ NO ──▶ SCANNED PAGE
                 │
                 ▼
         ┌─────────────────┐
         │ Render to Image │
         │ (250 DPI)       │
         └─────────────────┘
                 │
                 ▼
         ┌─────────────────┐
         │ Preprocess      │
         │ • Grayscale     │
         │ • Gaussian Blur │
         │ • Adaptive      │
         │   Threshold     │
         └─────────────────┘
                 │
                 ▼
         ┌─────────────────┐
         │ PaddleOCR       │
         │ (Primary)       │
         └─────────────────┘
                 │
                 ▼
         Confidence ≥ 0.70?
                 │
         ┌───────┴───────┐
         ▼               ▼
        YES             NO
         │               │
         ▼               ▼
    Use Paddle      ┌─────────────────┐
    Result          │ EasyOCR         │
                    │ (Fallback)      │
                    └─────────────────┘
                            │
                            ▼
                    Compare Confidence
                            │
                    ┌───────┴───────┐
                    ▼               ▼
                 Higher          Lower
                    │               │
                    ▼               ▼
               Use EasyOCR     Use Paddle
```

### Image Preprocessing Details

```python
def preprocess(image):
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 2. Reduce noise with Gaussian blur
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    
    # 3. Adaptive thresholding (handles varying lighting)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 4. Convert back to RGB for OCR models
    rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    return rgb
```

**Why each step:**
| Step | Purpose |
|------|---------|
| Grayscale | Reduces 3 channels → 1, faster processing |
| Gaussian Blur | Removes salt-and-pepper noise |
| Adaptive Threshold | Handles shadows, uneven lighting |
| Back to RGB | Most OCR models expect 3-channel input |

### Confidence Scoring

```python
# Per-page confidence = average of all text line confidences
confidences = [0.95, 0.87, 0.92, 0.78, 0.89]
avg_confidence = sum(confidences) / len(confidences)  # 0.882

# Document-level flags
warning_flag = avg_confidence < 0.80
low_confidence_blocks = count(conf < 0.60)
```

---

## 7. Vector Databases & Embeddings {#vector-db}

### What is a Vector Database?

```
Traditional DB          Vector DB
─────────────          ──────────
Rows & Columns         Vectors (lists of numbers)
Exact Match            Similarity Search
WHERE name = 'John'    "Find vectors closest to [0.1, -0.3, 0.7...]"
```

### Embeddings: Text → Numbers

```
Text: "The cat sits on the mat"
              │
              ▼
    Embedding Model
    (nomic-embed-text)
              │
              ▼
    Vector: [0.12, -0.45, 0.78, 0.33, -0.11, ...]  (768 dimensions)
```

**Key Property:** Similar text → Similar vectors (close in vector space)

```
"The cat sits"     → [0.12, -0.45, 0.78, ...]
"The dog runs"     → [0.15, -0.42, 0.75, ...]  (close - both animals)
"The stock market" → [-0.88, 0.92, -0.11, ...] (far - different topic)
```

### Pinecone Configuration in This System

```json
{
  "index_name": "rag",
  "dimension": 768,              // nomic-embed-text output
  "metric": "cosine",            // Similarity metric
  "pods": 1,                     // Can scale
  "replicas": 1                  // High availability
}
```

### Chunking Strategy

```python
# RecursiveCharacterTextSplitter
chunk_size = 1000        # Target chunk size
chunk_overlap = 200      # Overlap for context continuity
separators = ["\n\n", "\n", " ", ""]  # Recursive splitting
```

**Why overlap?**
```
Chunk 1: "The quick brown fox jumps over the lazy dog. The dog..."
Chunk 2: "...jumps over the lazy dog. The dog barked at the cat."
                        ↑
                  Overlap preserves context
```

### Metadata Stored Per Chunk

```json
{
  "pageContent": "The quick brown fox...",
  "metadata": {
    "filename": "animals.pdf",
    "extension": "pdf",
    "document_type": "scanned_pdf",
    "pages": 15,
    "ocr": true,
    "ocr_engine": "paddle",
    "confidence": 0.92,
    "blocks": 47,
    "processing_time": 3.4,
    "source": "Google Drive",
    "created_at": "2026-09-05T10:30:00Z",
    "document_category": "image",
    "status": "processed"
  }
}
```

---

## 8. The Retrieval Process {#retrieval}

### Two-Stage Retrieval

```
User Question: "What is the CEO's name?"
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 1: VECTOR SEARCH (Pinecone)                      │
│ • Embed question → [0.12, -0.45, 0.78, ...]            │
│ • Cosine similarity against all vectors                │
│ • Return top-20 candidates                             │
│ • Fast but approximate                                 │
└────────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 2: RERANKING (Cohere)                            │
│ • Cross-encoder: question + each chunk → relevance     │
│ • More accurate but slower                             │
│ • Reorder top-20 → top-5 most relevant                 │
└────────────────────────────────────────────────────────┘
                    │
                    ▼
         TOP 5 CHUNKS TO LLM
```

### Why Rerank?

| Stage | Speed | Accuracy | Purpose |
|-------|-------|----------|---------|
| Vector Search | ~10ms | Good | Broad recall |
| Reranker | ~200ms | Excellent | Precision |

**Without rerank:** Might get chunks about "CEO salary" when asking "CEO name"
**With rerank:** Understands semantic nuance, ranks "CEO name: John" higher

---

## 9. Generation & Answer Synthesis {#generation}

### System Prompt Engineering

The agent receives this system prompt (truncated for readability):

```xml
<AgentInstructions>
    <Role>
        <Name>Knowledge Assistant</Name>
        <Description>
            You are an advanced retrieval-based document assistant.
            You answer questions using information retrieved from uploaded documents.
            You do not possess personal knowledge.
            You answer only using retrieved context.
        </Description>
    </Role>

    <Goal>
        <Primary>Answer questions accurately using retrieved documents.</Primary>
        <Secondary>Produce concise and reliable answers.</Secondary>
        <Secondary>Cite document metadata whenever available.</Secondary>
        <Secondary>Prevent hallucinations.</Secondary>
    </Goal>

    <Instructions>
        <Instruction>Always use Pinecone_Vector_Store before answering any question.</Instruction>
        <Instruction>Never answer from memory.</Instruction>
        <Instruction>Never answer using prior knowledge.</Instruction>
        <Instruction>Never skip document retrieval.</Instruction>
        <Instruction>Never invent information.</Instruction>
        <Instruction>Never invent filenames.</Instruction>
        <Instruction>Never invent page numbers.</Instruction>
        <Instruction>If Pinecone_Vector_Store returns no results, respond exactly: "I don't know based on uploaded documents."</Instruction>
        <Instruction>When citing sources, use the format: [filename, page X]</Instruction>
        <Instruction>If confidence < 0.8, mention uncertainty.</Instruction>
    </Instructions>
</AgentInstructions>
```

### Critical Instructions Explained

| Instruction | Why It Matters |
|-------------|----------------|
| "Never answer from memory" | Prevents hallucination |
| "Never invent filenames" | Ensures verifiable citations |
| "If no results → 'I don't know'" | Honest about limitations |
| "Cite as [filename, page X]" | Traceable answers |
| "Mention uncertainty if confidence < 0.8" | Calibrated confidence |

### Memory Buffer

```json
{
  "type": "memoryBufferWindow",
  "window_size": 5  // Keeps last 5 exchanges
}
```

**Purpose:** Maintains conversation context for follow-up questions.

---

## 10. N8N Workflow Architecture {#n8n-workflow}

### Workflow 1: File Router (Ingestion)

**Trigger:** Google Drive file creation
**Complexity:** 80+ nodes, 200+ connections

Key architectural patterns:

#### Pattern 1: Merge by Position
```json
{
  "type": "n8n-nodes-base.merge",
  "mode": "combine",
  "combineBy": "combineByPosition"
}
```
**Use case:** Combine OCR results with original metadata preserving order.

#### Pattern 2: Conditional Routing (Switch)
```json
{
  "type": "n8n-nodes-base.switch",
  "rules": {
    "values": [
      {"conditions": [...], "outputKey": "IMAGES"},
      {"conditions": [...], "outputKey": "PDFS"},
      ...
    ]
  }
}
```
**Use case:** Route 20+ file types to appropriate processors.

#### Pattern 3: Code Nodes for Transformation
```javascript
// Normalize metadata from different sources
return items.map(item => {
  const meta = item.json.metadata?.metadata || item.json.metadata || {};
  item.json.metadata = {
    filename: meta.filename,
    extension: meta.extension,
    document_type: meta.document_type,
    pages: meta.pages,
    ocr: meta.ocr,
    confidence: meta.confidence
  };
  return item;
});
```

#### Pattern 4: Confidence Gating
```json
{
  "type": "n8n-nodes-base.if",
  "conditions": {
    "conditions": [{
      "leftValue": "={{ $json.metadata.confidence }}",
      "rightValue": 0.6,
      "operation": "gt"
    }]
  }
}
```
**Use case:** Flag low-quality OCR for human review.

### Workflow 2: Knowledge Assistant (Query)

**Trigger:** Chat message (webhook)
**Complexity:** 7 nodes, clean LangChain integration

Key patterns:

#### Pattern 1: Tool-Enabled Agent
```json
{
  "type": "@n8n/n8n-nodes-langchain.agent",
  "parameters": {
    "tools": ["Pinecone Vector Store"],
    "systemMessage": "..."
  }
}
```

#### Pattern 2: Retrieval as Tool
```json
{
  "type": "@n8n/n8n-nodes-langchain.vectorStorePinecone",
  "parameters": {
    "mode": "retrieve-as-tool",
    "useReranker": true
  }
}
```

---

## 11. Production Considerations {#production}

### Scaling the OCR API

```yaml
# docker-compose.yml for production
services:
  ocr-api:
    build: .
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4G
          cpus: '2'
    environment:
      - WORKERS=4
```

### Monitoring Endpoints

Add to each FastAPI app:
```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": True,
        "gpu_available": torch.cuda.is_available()
    }

@app.get("/metrics")
async def metrics():
    return {
        "requests_total": REQUEST_COUNT,
        "avg_latency_ms": AVG_LATENCY,
        "error_rate": ERROR_RATE
    }
```

### Pinecone Optimization

| Setting | Recommendation |
|---------|----------------|
| Namespace | Use per-client or per-project |
| Metadata filtering | Filter by `document_type`, `source` |
| Batch upserts | 100 vectors per batch |
| Pod type | `p1.x1` for dev, `p2.x4` for prod |

### Cost Optimization

| Component | Cost Driver | Optimization |
|-----------|-------------|--------------|
| Pinecone | Storage + queries | Namespace, filter, batch |
| Cohere Rerank | API calls | Only rerank top-20 |
| Ollama | GPU memory | Quantized models (4-bit) |
| OCR API | CPU/GPU | Batch requests, cache |

---

## 12. How to Extend & Customize {#extend}

### Add New File Type Support

1. **Add to Switch node in File Router:**
```json
{
  "conditions": [{
    "leftValue": "={{$binary.data.fileExtension }}",
    "rightValue": "pptx",
    "operation": "equals"
  }],
  "outputKey": "TEXT_FILES"
}
```

2. **Ensure Extract from File supports it** (n8n's Extract from File node supports pptx natively)

### Add New OCR Language

```python
# In pdf_ocr_hybrid.py
easy = easyocr.Reader(['en', 'hi', 'fr', 'de', 'es'])  # Add languages

# In paddle_test.py / pdf_ocr_fast_api.py
paddle = PaddleOCR(lang='en')  # PaddleOCR supports: en, ch, fr, de, ko, ja, etc.
```

### Change Embedding Model

```json
// In workflow templates
"parameters": {
  "model": "mxbai-embed-large:latest"  // Instead of nomic-embed-text
}
```
**Note:** Must match Pinecone index dimension (768 for nomic, 1024 for mxbai)

### Change LLM

```json
// In Knowledge_Assistant template
"parameters": {
  "model": "llama3.1:70b",  // Larger model
  "options": {
    "numPredict": 1024,
    "temperature": 0.1
  }
}
```

### Add Hybrid Search (Keyword + Vector)

```python
# Pinecone supports hybrid search
from pinecone import Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("rag")

# Sparse + dense vectors
results = index.query(
    vector=dense_vector,
    sparse_vector=sparse_vector,
    top_k=10,
    include_metadata=True
)
```

---

## 13. Troubleshooting Guide {#troubleshooting}

### Common Issues & Solutions

#### OCR Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Empty text from PDF | Digital PDF but text layer corrupted | Increase threshold from 100 to 50 chars |
| Low confidence (<0.6) | Poor scan quality, handwriting | Increase DPI to 400, try server model |
| Hindi text not recognized | EasyOCR not loaded with 'hi' | Add 'hi' to Reader languages |
| Memory error on large PDF | Loading all pages at once | Process page by page (already implemented) |

#### Pinecone Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Index not found" | Wrong index name | Check `pineconeIndex.value` = "rag" |
| Dimension mismatch | Embedding model changed | Recreate index with new dimension |
| Slow queries | Large index, no filtering | Add metadata filters, use namespaces |
| Reranker timeout | Cohere API slow | Increase timeout, reduce top-K |

#### N8N Workflow Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Workflow not triggering | Google Drive webhook expired | Re-authenticate, recreate trigger |
| Credential errors | Placeholder not replaced | Fill all `{{PLACEHOLDERS}}` |
| Memory exceeded | Large PDF processing | Increase N8N memory, process in batches |
| Infinite loop | Agent keeps calling tool | Check `maxIterations: 3`, tool description |

#### Ollama Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Model not found | Not pulled | `ollama pull nomic-embed-text` |
| Slow embeddings | CPU only | Use GPU, smaller model |
| Connection refused | Ollama not running | `ollama serve` |

### Debug Commands

```bash
# Test OCR API
curl -X POST http://localhost:8000/ocr \
  -F "file=@test_image.jpg"

# Test PDF OCR
curl -X POST http://localhost:8000/pdf_fast_ocr \
  -F "file=@document.pdf"

# Check Pinecone index stats
curl -X POST https://<index>.pinecone.io/describe_index_stats \
  -H "Api-Key: $PINECONE_API_KEY"

# Test Ollama
curl http://localhost:11434/api/embeddings \
  -d '{"model": "nomic-embed-text", "prompt": "test"}'

# View N8N logs
docker logs n8n  # or check ~/.n8n/logs
```

---

## Appendix: File Reference

### Python Files
| File | Purpose | Entry Point |
|------|---------|-------------|
| `pdf_ocr_hybrid.py` | Production hybrid OCR | `uvicorn pdf_ocr_hybrid:app` |
| `pdf_ocr_fast_api.py` | Speed-optimized adaptive OCR | `uvicorn pdf_ocr_fast_api:app` |
| `pdf_ocr_api.py` | Simple baseline OCR | `uvicorn pdf_ocr_api:app` |
| `ocr_api.py` | Single image OCR | `uvicorn ocr_api:app` |
| `ocr.py` | PaddleOCR quick test | `python ocr.py` |
| `paddle_test.py` | Direct PDF prediction test | `python paddle_test.py` |

### Workflow Templates (Parameterized)
| File | Purpose | Key Placeholders |
|------|---------|------------------|
| `workflows/Knowledge_Assistant.template.json` | RAG chat agent | `PINECONE_CREDENTIAL_ID`, `OLLAMA_CREDENTIAL_ID`, `COHERE_CREDENTIAL_ID`, `WEBHOOK_ID` |
| `workflows/A_File_Router_Workflow.template.json` | Auto-ingestion pipeline | `GOOGLE_DRIVE_FOLDER_ID`, `GOOGLE_DRIVE_CREDENTIAL_ID`, `PINECONE_CREDENTIAL_ID`, `OLLAMA_CREDENTIAL_ID`, `OCR_API_URL` |

### Config Files
| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment variables template |
| `.gitignore` | Excludes venv, cache, secrets |
| `README.md` | Quick start guide |

---

## Learning Path for RAG Beginners

### Week 1: Foundations
- [ ] Read [What is RAG?](#what-is-rag) section
- [ ] Understand embeddings vs keywords
- [ ] Run `python ocr.py` to verify PaddleOCR works

### Week 2: Ingestion Pipeline
- [ ] Deploy `pdf_ocr_hybrid.py` locally
- [ ] Test with sample PDFs (digital + scanned)
- [ ] Examine `A_File_Router_Workflow.template.json` structure

### Week 3: Query Pipeline
- [ ] Set up Pinecone index "rag"
- [ ] Pull Ollama models: `ollama pull nomic-embed-text llama3.1:8b`
- [ ] Import Knowledge Assistant template to N8N

### Week 4: Production Hardening
- [ ] Add monitoring, logging
- [ ] Configure Cohere reranker
- [ ] Load test with 1000+ documents
- [ ] Set up CI/CD for workflow templates

---

## Glossary

| Term | Definition |
|------|------------|
| **Chunk** | A piece of document text (typically 500-1000 chars) |
| **Embedding** | Numerical vector representation of text |
| **Vector DB** | Database optimized for similarity search on vectors |
| **Reranker** | Model that reorders search results by relevance |
| **Hallucination** | LLM generating false information |
| **Grounding** | Constraining LLM output to retrieved facts |
| **Top-K** | Number of results retrieved (e.g., K=5) |
| **Cosine Similarity** | Measure of vector alignment (1=identical, 0=orthogonal) |
| **OCR** | Optical Character Recognition (image → text) |
| **DPI** | Dots Per Inch (image resolution for OCR) |
| **Confidence** | OCR model's certainty (0.0-1.0) |

---

## Final Notes

This system represents a **production-grade RAG implementation** with:

✅ **Automated ingestion** from Google Drive  
✅ **Multi-engine OCR** with confidence scoring  
✅ **Hybrid search** (vector + reranker)  
✅ **Citation-enforced generation**  
✅ **Parameterized templates** for secure deployment  
✅ **Comprehensive metadata** for filtering/analytics  

The templates are designed to be **forked and customized** - replace placeholders, adjust chunk sizes, swap models, add file types. The architecture supports scaling from hundreds to millions of documents.

---

*Generated as part of N8N RAG Agent deployment package*  
*For questions, check the GitHub repository issues or N8N community forums*