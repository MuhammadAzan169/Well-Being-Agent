# Well-Being-Agent

## Overview

The Well-Being-Agent is an AI-powered breast cancer support system that provides compassionate, evidence-based information and emotional support to breast cancer patients and their families. The system uses advanced Retrieval-Augmented Generation (RAG) technology to deliver accurate, personalized responses in both English and Urdu, with support for both text and voice queries.

## Project Structure

### Core Files

- **`app.py`** - Main FastAPI web server that handles HTTP requests, serves the frontend, and orchestrates the RAG system. Includes endpoints for:
  - `/` - Serves the main web interface
  - `/ask-query` - Processes text queries about breast cancer
  - `/voice-query` - Processes voice queries (speech-to-text, then text response)
  - `/predefined-questions` - Returns curated question suggestions
  - `/health` - System health check
  - `/info` - System information

- **`Agent.py`** - Contains the core AI logic including:
  - `BreastCancerRAGSystem` class - Main RAG implementation
  - Response caching system for performance
  - Language detection and text processing
  - Emotional support integration
  - Conversation logging

- **`audio_processor.py`** - Handles voice input processing:
  - Speech-to-text conversion using Whisper/OpenAI
  - Audio file management and cleanup
  - Language detection from speech

- **`Index.py`** - Builds and manages the vector index for document retrieval from the breast cancer dataset

- **`language_utils.py`** - Utility functions for language processing, text cleaning, and Urdu text normalization

### Data Files

- **`DataSet/breast_cancer.json`** - Comprehensive dataset containing breast cancer information, Q&A pairs, treatment details, and medical knowledge

- **`DataSet/Question.json`** - Additional question templates and predefined queries

- **`conversations.json`** - Logs of user interactions for analysis and improvement

### Frontend Files

- **`index.html`** - Main web interface with modern UI, supporting both English and Urdu text input, voice recording, and quick question buttons

- **`styles.css`** - Styling for the web interface with responsive design and Urdu font support

- **`script.js`** - Frontend JavaScript handling user interactions, API calls, voice recording, and UI updates

### Configuration and Dependencies

- **`requirements.txt`** - Python dependencies including FastAPI, LlamaIndex, OpenAI, Whisper, and other ML libraries

- **`Dockerfile`** - Containerization setup for deployment

- **`.env`** - Environment variables (API keys, configuration)

### Index Store

- **`cancer_index_store/`** - Vector database for efficient document retrieval:
  - `docstore.json` - Document storage
  - `graph_store.json` - Relationship graphs
  - `image__vector_store.json` - Vector embeddings
  - `index_store.json` - Main index data

### Cache and Temporary Files

- **`cache/`** - Response cache to improve performance and reduce API calls

- **`static/audio/`** - Temporary storage for audio files (auto-cleaned)

- **`__pycache__/`** - Python bytecode cache

## How It Works

### 1. System Initialization
- On startup, `app.py` loads the vector index from `cancer_index_store/`
- Initializes the `BreastCancerRAGSystem` with the index and retriever
- Sets up FastAPI server with CORS and static file serving

### 2. Query Processing
- User submits query via text input or voice recording
- For voice queries: `audio_processor.py` converts speech to text using Whisper
- Language detection determines if query is in English or Urdu
- `BreastCancerRAGSystem` retrieves relevant chunks from the vector store
- LLM (via OpenAI API) generates contextual response
- Response is enhanced with emotional support and formatted appropriately

### 3. Response Generation
- System uses cached responses when available to reduce latency
- Applies language-specific formatting and text cleaning
- Adds compassionate, supportive language naturally
- Logs conversation for quality improvement

### 4. Frontend Interaction
- Modern web interface with animated elements
- Support for both English and Urdu input/output
- Voice recording with real-time feedback
- Quick question buttons for common concerns
- Responsive design for mobile and desktop

## Key Features

- **Multilingual Support**: Full English and Urdu language support with proper text rendering
- **Voice Input**: Speech-to-text capability for accessibility
- **Emotional Support**: AI responses include compassionate, supportive language
- **Evidence-Based**: Responses grounded in medical knowledge from curated datasets
- **Caching**: Response caching for improved performance
- **Conversation Logging**: Tracks interactions for system improvement
- **Predefined Questions**: Curated question suggestions for common patient concerns

## Technology Stack

- **Backend**: FastAPI (Python web framework)
- **AI/ML**: 
  - LlamaIndex for RAG implementation
  - OpenAI GPT for response generation
  - Whisper for speech-to-text
  - Sentence Transformers for embeddings
- **Frontend**: HTML5, CSS3, JavaScript
- **Deployment**: Docker containerization
- **Vector Store**: Local JSON-based vector storage

## Usage

1. **Local Development**:
   ```bash
   pip install -r requirements.txt
   python app.py
   ```

2. **Docker Deployment**:
   ```bash
   docker build -t well-being-agent .
   docker run -p 7860:7860 well-being-agent
   ```

3. **Access**: Open browser to `http://localhost:7860`

## API Endpoints

- `GET /` - Main web interface
- `POST /ask-query` - Text query processing
- `POST /voice-query` - Voice query processing
- `GET /predefined-questions` - Get suggested questions
- `GET /health` - System health check
- `GET /info` - System information

## Configuration

Environment variables in `.env`:
- OpenAI API key for LLM responses
- Model configuration (defaults to Llama 3.1 70B)
- Port configuration (default 7860)

## Data Sources

The system uses a comprehensive breast cancer knowledge base including:
- Medical diagnosis procedures
- Treatment options and timelines
- Recovery guidance
- Emotional support strategies
- Nutritional advice
- Exercise recommendations during treatment

All information is evidence-based and focused on patient support and well-being.</content>
<filePath>c:\Users\muham\OneDrive\Desktop\Well-Being-Agent\README.md