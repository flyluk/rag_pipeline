# RAG Pipeline

A comprehensive RAG (Retrieval-Augmented Generation) pipeline system that processes documents, categorizes them using AI, and stores them in PostgreSQL with pgvector for efficient similarity search.

## Overview

The system consists of main components:
- **PostgresBulkIngest**: Core document processing with PostgreSQL/pgvector backend
- **RAGSystem**: AI integration and document analysis
- **Streamlit Interface**: Web UI for querying and browsing documents

## Features

- Document processing (PDF, DOCX, TXT)
- AI-powered document categorization and tagging
- PostgreSQL with pgvector for vector storage
- Automatic document summarization
- Interruption recovery and failed file tracking
- Real-time processing metrics and timing
- Streamlit web interface for document querying
- Docker Compose setup with all services

## Components

### PostgresBulkIngest (`chroma_bulk_ingest.py`)

Core class that handles:
- Document loading and processing
- AI-powered categorization using Ollama LLM
- PostgreSQL/pgvector storage
- Interruption recovery and retry logic

#### Key Methods:
- `ingest_documents()`: Process and store documents with timing metrics
- `file_exists_in_db()`: Check for duplicate files
- `get_files_by_category()`: Retrieve files grouped by category
- `extract_metadata()`: AI-powered document analysis

### Streamlit Interface (`streamlit_app.py`)

Web interface providing:
- Document querying with category filtering
- File browsing by category
- Real-time search results
- Source and metadata display

## Usage

### Setup with Docker Compose

1. Start all services:
```bash
docker-compose up -d
```

2. Setup database (first time only):
```bash
./setup_db.sh
```

### Manual Setup

1. Install dependencies:
```bash
pip install langchain langchain-postgres psycopg2-binary
```

2. Setup PostgreSQL with pgvector:
```bash
./setup_db.sh
```

### Processing Documents

```bash
python chroma_bulk_ingest.py /path/to/documents
```

### Web Interface

```bash
streamlit run streamlit_app.py
```

### Basic Workflow

1. **Process Files**: Bulk ingest documents with AI analysis
2. **Vector Storage**: Documents stored in PostgreSQL with embeddings
3. **Query Interface**: Use Streamlit UI to search and browse documents

## Configuration

### PostgresBulkIngest Parameters

```python
ingest = PostgresBulkIngest(
    collection_name="documents",
    connection_string="postgresql://raguser:ragpassword@localhost:5432/vectordb",
    embedding_model="nomic-embed-text",
    llm_model="deepseek-r1:7b"
)
```

### Document Processing Flow

1. **File Detection** → Find supported documents
2. **Duplicate Check** → Skip already processed files
3. **Content Extraction** → Load document content
4. **AI Analysis** → Categorize and tag using LLM
5. **Vector Storage** → Store embeddings in PostgreSQL
6. **Progress Tracking** → Real-time metrics and failed file recovery

## Database Management

### Reset Database
```bash
./reset_db.sh
```

### Reindex for Performance
```bash
./reindex_db.sh
```

### Connection String
```
postgresql://raguser:ragpassword@localhost:5432/vectordb
```

## File Support

- **PDF**: PyPDFLoader
- **DOCX**: Docx2txtLoader  
- **TXT**: TextLoader with UTF-8 encoding

## AI Features

### Document Categorization
Extracts:
- Main category
- Key topics (list)
- Sentiment (positive/negative/neutral)
- Language detection

### Summarization
Generates concise summaries focusing on:
- Main points
- Key findings
- Important conclusions

## Features

### Processing Metrics
- Individual file processing time
- Running average time per file
- Total elapsed time
- Progress tracking with file counts

### Error Recovery
- Interruption handling (Ctrl+C)
- Failed file tracking in `failed_files.txt`
- Automatic reprocessing of failed files
- Duplicate detection and skipping

### Performance
- PostgreSQL with pgvector for fast similarity search
- Optimized indexing for vector operations
- Batch processing with real-time feedback

## Dependencies

- `langchain`: LLM integration
- `langchain-postgres`: PostgreSQL vector store
- `langchain-ollama`: Ollama integration
- `psycopg2-binary`: PostgreSQL adapter
- `streamlit`: Web interface
- `pgvector`: PostgreSQL vector extension