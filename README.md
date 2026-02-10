# Philippine Politician RAG Pipeline

RAG (Retrieval-Augmented Generation) pipeline for summarizing and querying Philippine government documents, laws, and politician information.

## Architecture

- **LLM:** OpenAI GPT-4o-mini
- **Embeddings:** OpenAI text-embedding-3-small
- **Vector DB:** Pinecone (hosted)
- **API:** FastAPI
- **Package Manager:** uv

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key
- Pinecone API key + index

### Install dependencies

```bash
uv sync
```

### Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Create Pinecone index

Create an index in the [Pinecone dashboard](https://app.pinecone.io/) with:
- **Dimensions:** 1536 (for text-embedding-3-small)
- **Metric:** cosine
- **Name:** ph-politician-rag (or whatever you set in .env)

### Download sample documents

```bash
uv run python scripts/download_samples.py
```

### Ingest documents into Pinecone

```bash
uv run python scripts/ingest.py
```

### Run the API

```bash
uv run uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### `GET /api/health`

Health check.

### `POST /api/query`

Query the document database.

**Request:**
```json
{
  "question": "What is the Cybercrime Prevention Act?",
  "max_results": 5
}
```

**Response:**
```json
{
  "answer": "The Cybercrime Prevention Act (Republic Act No. 10175)...",
  "sources": [
    {
      "content": "Section 1. Title...",
      "title": "Republic Act No. 10175 - Cybercrime Prevention Act of 2012",
      "source": "Official Gazette",
      "type": "law",
      "date": "2012-09-12"
    }
  ]
}
```

## Deployment

### Docker

```bash
docker build -t ph-politician-rag .
docker run -p 8000:8000 --env-file .env ph-politician-rag
```

### Railway / Render

1. Push to GitHub
2. Connect your repo in Railway or Render
3. Set environment variables (`OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX`)
4. Deploy — the `Procfile` handles the start command

## Project Structure

```
rag-pipeline/
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI app + endpoints
│   │   └── models.py        # Request/response models
│   └── rag/
│       ├── chain.py          # RAG query engine
│       └── prompts.py        # LLM prompt templates
├── scripts/
│   ├── download_samples.py   # Fetch sample docs
│   └── ingest.py             # Embed & upsert into Pinecone
├── data/raw/                  # Downloaded documents (gitignored)
├── pyproject.toml
├── Dockerfile
└── Procfile
```
