# 💛 WellBeing Agent — Breast Cancer Support System

> A RAG-based AI well-being assistant designed to support breast cancer patients during their treatment journey with empathetic, evidence-based information.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation Guide](#-installation-guide)
- [Running the System](#-running-the-system)
- [Voice Input Setup](#-voice-input-setup)
- [Folder Structure](#-folder-structure)
- [Language Support](#-language-support)
- [API Endpoints](#-api-endpoints)
- [Safety Disclaimer](#-safety-disclaimer)

---

## 🌟 Project Overview

**WellBeing Agent** is a Retrieval-Augmented Generation (RAG) powered AI assistant specifically designed to support breast cancer patients and their families. Unlike medical chatbots, this agent focuses exclusively on **well-being support** — providing empathetic answers, emotional comfort, lifestyle guidance, and practical coping strategies.

### What this agent does:
- ✅ Answers patient questions about symptoms, side effects, and recovery
- ✅ Provides emotional support and coping strategies
- ✅ Offers nutritional guidance during treatment
- ✅ Addresses concerns about body changes (e.g., after mastectomy)
- ✅ Supports multilingual queries (English, Urdu, Roman Urdu)
- ✅ Accepts voice input via Whisper Large v3

### What this agent does NOT do:
- ❌ Does **not** prescribe medications or treatments
- ❌ Does **not** replace professional medical advice
- ❌ Does **not** diagnose conditions
- ❌ Does **not** recommend stopping any treatment

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **RAG-Powered Q&A** | Retrieves answers from a curated breast cancer knowledge base and enhances them with LLM generation |
| **Breast Cancer Knowledge** | Comprehensive dataset covering symptoms, treatments, side effects, recovery, and well-being |
| **Emotional Support** | Detects emotional distress and adjusts response tone with empathy-first approach |
| **Nutrition Guidance** | Provides dietary suggestions and lifestyle tips during treatment |
| **Voice Input** | Whisper Large v3 for high-quality speech-to-text transcription |
| **Multilingual Support** | English + Urdu (script) + Roman Urdu (transliterated) |
| **Response Caching** | JSON-based cache with similarity matching for faster repeat queries and error resilience |
| **Safety Guardrails** | Crisis detection, off-topic filtering, dangerous advice prevention |
| **Source Citations** | Every answer includes relevant source references |
| **API Key Rotation** | Automatic failover between multiple API keys |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (HTML/JS/CSS)               │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │ Text Chat │  │ Voice Record │  │ Predefined Questions│   │
│  └─────┬────┘  └──────┬───────┘  └──────────┬──────────┘   │
└────────┼──────────────┼──────────────────────┼──────────────┘
         │              │                      │
         ▼              ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Server (app.py)                   │
│  ┌──────────┐  ┌────────────┐  ┌──────────────────────┐    │
│  │/ask-query│  │/voice-query│  │/predefined-questions │    │
│  └─────┬────┘  └──────┬─────┘  └──────────────────────┘    │
└────────┼──────────────┼─────────────────────────────────────┘
         │              │
         │              ▼
         │    ┌──────────────────┐
         │    │  Whisper Large v3 │ ← Speech-to-Text
         │    │ (audio_processor) │
         │    └────────┬─────────┘
         │             │
         ▼             ▼
┌─────────────────────────────────────────────────────────────┐
│              RAG Pipeline (Agent.py)                         │
│                                                             │
│  ┌───────────┐  ┌────────────┐  ┌──────────────────────┐   │
│  │ Language   │  │  Safety    │  │  Response Cache      │   │
│  │ Detection  │  │ Validator  │  │  (JSON + Similarity) │   │
│  └─────┬─────┘  └──────┬─────┘  └──────────┬───────────┘   │
│        ▼               ▼                    ▼               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Vector Retrieval (LlamaIndex)           │   │
│  │  ┌─────────────────┐  ┌──────────────────────────┐   │   │
│  │  │ HuggingFace     │  │ Vector Store              │   │   │
│  │  │ Embeddings      │  │ (cancer_index_store/)     │   │   │
│  │  └─────────────────┘  └──────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
│        │                                                    │
│        ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Prompt Builder → LLM (via API) → Post-Processing    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Component Details

| Component | File | Purpose |
|-----------|------|---------|
| **RAG Agent** | `Agent.py` | Core pipeline: language detection → safety → cache → retrieval → LLM → response |
| **Web Server** | `app.py` | FastAPI server with REST endpoints for text, voice, and health |
| **Audio Processor** | `audio_processor.py` | Whisper Large v3 speech-to-text transcription |
| **Language Utils** | `language_utils.py` | Language detection (English, Urdu script, Roman Urdu) and text normalization |
| **Safety Module** | `safety.py` | Crisis detection, off-topic filtering, dangerous advice prevention |
| **Index Builder** | `Index.py` | Builds the vector index from the breast cancer dataset |
| **Frontend** | `index.html`, `static/` | Chat interface with voice recording and bilingual support |
| **Knowledge Base** | `DataSet/` | Curated breast cancer support dataset (JSON) |

---

## 📦 Installation Guide

### Prerequisites

- Python 3.10+ (3.12 recommended)
- Git
- (Optional) CUDA-capable GPU for faster Whisper transcription

### Step 1: Clone the Repository

```bash
git clone https://github.com/MuhammadAzan169/Well-Being-Agent.git
cd Well-Being-Agent
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Whisper Large v3

Whisper Large v3 is included in the `transformers` package. The model will be automatically downloaded on first use (~3GB).

For GPU acceleration (recommended):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Step 5: Configure Environment

Create a `.env` file in the project root:

```env
# LLM Configuration
LLM_PROVIDER=openai          # or any OpenAI-compatible provider
LLM_MODEL=your-model-name
LLM_BASE_URL=https://your-api-base-url
LLM_API_KEY=your-api-key
LLM_API_KEY_2=optional-backup-key
LLM_API_KEY_3=optional-backup-key
LLM_MAX_TOKENS=1500
LLM_TEMPERATURE=0.3

# Embedding & Index
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
INDEX_PATH=cancer_index_store
DATASET_PATH=DataSet/breast_cancer_comprehensive.json
SIMILARITY_TOP_K=5

# Cache
CACHE_TTL_HOURS=24
CACHE_SIMILARITY_THRESHOLD=0.85

# Server
PORT=8000
```

### Step 6: Build the Vector Index

```bash
python Index.py
```

---

## 🚀 Running the System

### Option 1: Run with the FastAPI Server (recommended)

```bash
python app.py
```

Then open your browser at: **http://localhost:8000**

### Option 2: Run in CLI Mode

```bash
python Agent.py
```

This starts an interactive command-line interface for testing queries directly.

### Option 3: Run with Uvicorn (production)

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Option 4: Docker

```bash
docker build -t wellbeing-agent .
docker run -p 7860:7860 --env-file .env wellbeing-agent
```

---

## 🎤 Voice Input Setup

### How Whisper Works Locally

The system uses **OpenAI Whisper Large v3** via the Hugging Face `transformers` library for local speech-to-text transcription:

1. **Audio Recording**: The browser captures audio using the MediaRecorder API
2. **Base64 Encoding**: Audio is sent to the server as base64-encoded WebM
3. **Whisper Transcription**: The server decodes the audio and runs Whisper Large v3 locally
4. **Language Detection**: The system automatically detects the language from the transcription
5. **RAG Processing**: The transcribed text goes through the same RAG pipeline as text queries

### Voice Language Behavior

| User Input | Agent Response Language |
|------------|----------------------|
| English voice message | English |
| Urdu voice message | Urdu |
| English text | English |
| Urdu text (script) | Urdu |
| Roman Urdu text (e.g., "mera sir dard kar raha hai") | Urdu |

### GPU vs CPU

- **GPU (CUDA)**: Real-time transcription (~2-5 seconds)
- **CPU**: Slower transcription (~15-30 seconds per audio clip)

The model is loaded lazily on first voice query and cached in memory for subsequent requests.

---

## 📁 Folder Structure

```
Well-Being-Agent/
├── Agent.py                 # Core RAG pipeline and LLM integration
├── app.py                   # FastAPI web server
├── audio_processor.py       # Whisper Large v3 transcription pipeline
├── language_utils.py        # Language detection (EN/UR/Roman Urdu)
├── safety.py                # Safety guardrails and content validation
├── Index.py                 # Vector index builder
├── index.html               # Main web interface
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker deployment configuration
├── conversations.json       # Conversation log (auto-generated)
├── .env                     # Environment configuration (create manually)
│
├── static/
│   ├── script.js            # Frontend JavaScript
│   ├── styles.css           # Frontend CSS styles
│   └── audio/               # Temporary audio files (auto-cleaned)
│
├── cache/
│   └── response_cache.json  # Response cache (auto-generated)
│
├── cancer_index_store/      # Vector index (built by Index.py)
│   ├── default__vector_store.json
│   ├── docstore.json
│   ├── index_store.json
│   └── index_metadata.json
│
└── DataSet/
    ├── breast_cancer_comprehensive.json  # Primary knowledge base
    ├── breast_cancer.json                # Legacy dataset
    └── Question.json                     # Predefined questions
```

### Module Descriptions

| Module | Description |
|--------|-------------|
| `Agent.py` | The heart of the system. Contains the RAG pipeline: config, cache, retrieval, prompt engineering, LLM querying, and post-processing. |
| `app.py` | FastAPI server exposing REST endpoints for the web interface. Handles text queries, voice queries, and health checks. |
| `audio_processor.py` | Manages the Whisper Large v3 model lifecycle. Handles audio file transcription and temporary file cleanup. |
| `language_utils.py` | Comprehensive language detection supporting Urdu script, Roman Urdu (transliterated), and English. Also handles Urdu text normalization. |
| `safety.py` | Content safety guardrails: crisis detection (suicide/self-harm), off-topic filtering, dangerous medical advice detection, and disclaimer injection. |
| `Index.py` | Builds the vector search index from the JSON dataset using HuggingFace embeddings and LlamaIndex. Run once during setup. |

---

## 🌍 Language Support

The system supports three input modes:

### 1. English
Standard English text or voice input.
```
User: I feel very tired after radiation therapy
Agent: [Responds in English with empathy and guidance]
```

### 2. Urdu (Script)
Native Urdu text using Arabic script.
```
User: کیموتھراپی کے بعد تھکاوٹ کیسے کم کریں؟
Agent: [اردو میں جواب]
```

### 3. Roman Urdu (Transliterated)
Urdu written in English/Latin characters — common in Pakistan.
```
User: mera sir bohat dard kar raha hai chemotherapy ke baad
Agent: [اردو میں جواب — system detects Roman Urdu and responds in Urdu]
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the web interface |
| `POST` | `/ask-query` | Text query → RAG answer |
| `POST` | `/voice-query` | Voice audio (base64) → Whisper → RAG answer |
| `GET` | `/predefined-questions?language=english` | Get predefined FAQ questions |
| `GET` | `/health` | System health check |
| `GET` | `/info` | System information and features |

### Example Text Query

```bash
curl -X POST http://localhost:8000/ask-query \
  -H "Content-Type: application/json" \
  -d '{"message": "What foods should I eat during chemotherapy?", "language": "english"}'
```

### Response Format

```json
{
  "answer": "During chemotherapy, focusing on...",
  "sources": [
    {"topic": "Nutrition", "category": "lifestyle", "source": "Knowledge Base", "score": 0.85}
  ],
  "language": "english"
}
```

---

## ⚠️ Safety Disclaimer

> **IMPORTANT**: WellBeing Agent is an AI-powered well-being support tool. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment.

- 🏥 This agent **does not replace** medical professionals
- 💊 This agent **does not prescribe** treatments or medications
- 🩺 This agent **does not diagnose** medical conditions
- 💛 This agent **only provides** supportive guidance, educational information, and emotional reassurance

**Always consult your healthcare team** for medical decisions. If you are in crisis, please contact emergency services or a crisis hotline immediately.

---

## 📄 License

This project is for educational and supportive purposes. See the repository for license details.

---

<p align="center">
  Made with 💛 for breast cancer patients and their families
</p>
