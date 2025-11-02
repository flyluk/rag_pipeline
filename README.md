# RAG Pipeline

AI-powered document processing pipeline with PostgreSQL/pgvector storage and Open WebUI integration.

## Features

- **Document Processing**: PDF, DOCX, TXT with AI categorization
- **Vector Storage**: PostgreSQL with pgvector for similarity search
- **Web Interfaces**: Streamlit UI and Open WebUI integration
- **AI Analysis**: Document categorization, tagging, and summarization
- **Error Recovery**: Interruption handling and failed file tracking
- **Docker Setup**: Complete containerized environment

## Quick Start

### Docker Setup (Recommended)

```bash
# Start default services (Open WebUI + Ollama)
docker-compose up -d

# Start with specific services
ENABLE_POSTGRES=enabled ENABLE_DOCLING=enabled docker-compose up -d

# Or use .env file
echo "ENABLE_POSTGRES=enabled" >> .env
echo "ENABLE_DOCLING=enabled" >> .env
docker-compose up -d

# Setup database (first time)
./setup_db.sh
```

### Access Services
- **Open WebUI**: http://localhost:3000
- **Streamlit**: http://localhost:8000 (after running `streamlit run streamlit_app.py`)
- **PostgreSQL**: localhost:5432
- **Docling**: http://localhost:5001

## Usage

### Process Documents

**PostgreSQL Backend:**
```bash
python chroma_bulk_ingest.py /path/to/documents
```

**Open WebUI Integration:**
```bash
# AI categorization
python bulk_upload.py /path/to/documents --api-key YOUR_API_KEY

# Fixed category (faster)
python bulk_upload.py /path/to/documents --api-key YOUR_API_KEY --category "Documentation"

# API key from file
python bulk_upload.py /path/to/documents --api-key-file ~/.openwebui_api_key
```

### Web Interface
```bash
streamlit run streamlit_app.py
```

## Components

| Component | Purpose | Key Features |
|-----------|---------|-------------|
| **PostgresBulkIngest** | Core document processing | AI categorization, vector storage, error recovery |
| **BulkUpload** | Open WebUI integration | Bulk upload, knowledge base creation, progress tracking |
| **Streamlit App** | Web interface | Document search, category filtering, metadata display |
| **Docker Services** | Infrastructure | PostgreSQL, Ollama, Open WebUI, Docling |

## Configuration

### Environment Variables

**Service Control:**
```bash
ENABLE_OPENWEBUI=enabled    # Default: enabled
ENABLE_OLLAMA=enabled       # Default: enabled
ENABLE_POSTGRES=enabled     # Default: disabled
ENABLE_DOCLING=enabled      # Default: disabled
ENABLE_VLLM=enabled         # Default: disabled
ENABLE_DEVELOPMENT=enabled  # Default: disabled
```

**Application:**
```bash
DATABASE_URL=postgresql://raguser:ragpassword@postgres:5432/vectordb
DOCLING_URL=http://docling:5001
OPENWEBUI_URL=http://open-webui:8080
OLLAMA_API_URL=http://ollama:11434
HF_TOKEN=your_huggingface_token  # For vLLM service
```

### Default Models
- **Embedding**: nomic-embed-text
- **LLM**: deepseek-r1:7b
- **Supported Files**: PDF, DOCX, TXT

## AI Features

- **Categorization**: Automatic document classification with topics and sentiment
- **Summarization**: Key points and findings extraction
- **Language Detection**: Multi-language support
- **Vector Search**: Semantic similarity matching

## Database Management

```bash
# Reset database
./reset_db.sh

# Reindex for performance
./reindex_db.sh
```

## Error Recovery

- Interruption handling (Ctrl+C)
- Failed file tracking and reprocessing
- Duplicate detection
- Progress metrics and timing

## Dependencies

**Core:**
- langchain, langchain-postgres, langchain-ollama
- psycopg2-binary, pgvector
- streamlit, requests

**Services:**
- PostgreSQL with pgvector
- Ollama LLM server
- Open WebUI
- Docling document processing