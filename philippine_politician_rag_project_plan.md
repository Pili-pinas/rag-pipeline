# Philippine Politician LLM Project Plan
## Document Summarization & Analysis System

**Project Goal:** Build a RAG (Retrieval-Augmented Generation) system to summarize Philippine government documents, laws, and politician achievements/issues.

**Last Updated:** February 10, 2026

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technology Stack](#technology-stack)
3. [Data Sources](#data-sources)
4. [Project Architecture](#project-architecture)
5. [Implementation Phases](#implementation-phases)
6. [Code Examples & Setup](#code-examples--setup)
7. [Repository Structure](#repository-structure)
8. [Deployment Strategy](#deployment-strategy)
9. [Budget Estimates](#budget-estimates)
10. [Next Steps & Resources](#next-steps--resources)

---

## Project Overview

### What is RAG?
RAG (Retrieval-Augmented Generation) combines document retrieval with LLM generation:
1. **Store** documents in a vector database
2. **Retrieve** relevant chunks when user asks a question
3. **Generate** summaries using an LLM with the retrieved context

### Why RAG Instead of Fine-tuning?
- ✅ No expensive training required
- ✅ Easy to update with new documents
- ✅ More accurate for factual information
- ✅ Can cite specific sources
- ✅ Much cheaper to maintain ($0-20/month vs $100s for fine-tuning)

### Use Cases
- Summarize politician achievements and issues
- Search Philippine laws and regulations
- Compare politician voting records
- Track legislative changes over time
- Answer questions about government policies

---

## Technology Stack

### Core Components

#### LLM Options
**Option A: Free Local (Recommended to Start)**
- **Model:** Llama 3.2 (3B or 8B)
- **Platform:** Ollama (runs on laptop)
- **Cost:** $0
- **Pros:** Completely free, private, fast iteration
- **Cons:** Quality depends on hardware

**Option B: Paid API (Better Quality)**
- **Models:** 
  - OpenAI GPT-4o-mini (~$0.15 per million input tokens)
  - Claude Haiku (~$0.25 per million input tokens)
- **Cost:** ~$5-20/month depending on usage
- **Pros:** Better quality, no hardware requirements
- **Cons:** Ongoing costs, requires API key

#### Vector Database
**Recommended:** ChromaDB (self-hosted, free)
- Easy to set up locally
- Good for development
- Free forever

**Alternatives (with free tiers):**
- Pinecone (100k vectors free)
- Qdrant Cloud
- Weaviate

#### Framework
**LangChain** - Python library for building RAG applications
- Handles document loading, splitting, embeddings
- Integrates with most LLMs and vector databases
- Large community and documentation

**Alternative:** LlamaIndex (similar functionality)

#### Embeddings
**sentence-transformers/all-MiniLM-L6-v2**
- Free, runs locally
- Good quality for English text
- Fast processing

**For multilingual (English + Filipino):**
- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

---

## Data Sources

### Official Philippine Government Sources

#### Primary Sources
1. **Official Gazette** (officialgazette.gov.ph)
   - All laws, executive orders, proclamations
   - Presidential speeches and statements
   - Historical government documents

2. **Senate of the Philippines** (senate.gov.ph)
   - Bills and resolutions
   - Senator profiles and voting records
   - Committee reports

3. **House of Representatives** (congress.gov.ph)
   - Congressional records
   - Representative profiles
   - Legislative documents

4. **Supreme Court E-Library** (elibrary.judiciary.gov.ph)
   - Case law and decisions
   - Legal precedents

5. **Commission on Elections** (comelec.gov.ph)
   - Candidate information
   - Election results
   - Political party data

6. **SALN (Statement of Assets, Liabilities and Net Worth)**
   - Public official financial disclosures
   - Various sources (news sites, advocacy groups)

#### News Archives (for achievements/issues)
- Rappler (rappler.com)
- Philippine Daily Inquirer (inquirer.net)
- Philippine Star (philstar.com)
- Manila Bulletin (mb.com.ph)
- GMA News (gmanetwork.com)

### Data Collection Strategy

#### Web Scraping Tools
```python
# For HTML content
import requests
from bs4 import BeautifulSoup

# For PDFs (common in gov sites)
import pypdf2
import pdfplumber
```

#### Important Considerations
- ⚠️ Check robots.txt and terms of service before scraping
- ⚠️ Rate limit your requests (1-2 seconds between requests)
- ⚠️ Store raw documents for later re-processing
- ⚠️ Track document metadata (source, date, type, politician name)

#### Data Organization
```
data/
├── raw/                    # Original downloaded files
│   ├── laws/
│   ├── politician_profiles/
│   └── news_articles/
├── processed/              # Cleaned and formatted
└── metadata.json           # Document index with metadata
```

---

## Project Architecture

### Repository Structure Decision

**Recommended: Monorepo (Start Simple)**

```
ph-politician-ai/
├── backend/                        # RAG API
│   ├── src/
│   │   ├── data_ingestion/        # Scrapers, document processors
│   │   │   ├── scrapers/
│   │   │   │   ├── official_gazette.py
│   │   │   │   ├── senate.py
│   │   │   │   └── news_sites.py
│   │   │   ├── processors/
│   │   │   │   ├── pdf_processor.py
│   │   │   │   └── html_processor.py
│   │   │   └── ingestion.py
│   │   ├── embeddings/             # Vector DB management
│   │   │   ├── create_embeddings.py
│   │   │   └── vector_store.py
│   │   ├── retrieval/              # RAG logic
│   │   │   ├── rag_chain.py
│   │   │   └── prompts.py
│   │   └── api/                    # FastAPI endpoints
│   │       └── main.py
│   ├── data/                       # Document storage
│   │   ├── raw/
│   │   └── processed/
│   ├── vector_db/                  # ChromaDB storage
│   ├── requirements.txt
│   └── README.md
│
├── frontend/                       # Streamlit UI
│   ├── app.py
│   ├── components/
│   │   ├── search_interface.py
│   │   └── results_display.py
│   ├── requirements.txt
│   └── README.md
│
├── docker-compose.yml              # Run both together
├── .gitignore
└── README.md                       # Main project docs
```

**When to Split into Separate Repos:**
- When you want to deploy them independently
- When you have a stable API contract
- When you want to open-source one but not the other
- When you have different teams working on each

---

## Implementation Phases

### Phase 1: Setup & Proof of Concept (Week 1-2)

**Goals:**
- Set up development environment
- Collect 50-100 sample documents
- Build basic RAG pipeline
- Test with simple queries

**Tasks:**

1. **Environment Setup**
```bash
# Create project directory
mkdir ph-politician-ai
cd ph-politician-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install langchain chromadb openai sentence-transformers
pip install beautifulsoup4 requests pypdf2 pdfplumber
pip install streamlit fastapi uvicorn
```

2. **Data Collection**
- Manually download 50-100 sample documents from Official Gazette
- Mix of laws, executive orders, and politician profiles
- Store in `backend/data/raw/`

3. **Basic RAG Pipeline**
- Create embeddings from sample documents
- Set up ChromaDB
- Build simple query interface
- Test retrieval quality

4. **Success Criteria**
- Can successfully retrieve relevant documents
- LLM generates coherent summaries
- Basic Streamlit interface works

### Phase 2: Data Pipeline & Expansion (Week 3-4)

**Goals:**
- Automate data collection
- Expand document collection to 500+ documents
- Improve document processing
- Add metadata tracking

**Tasks:**

1. **Build Web Scrapers**
- Official Gazette scraper
- Senate website scraper
- News site scrapers (Rappler, Inquirer)
- PDF download and processing

2. **Document Processing Pipeline**
- Clean and normalize text
- Extract metadata (date, author, type)
- Handle mixed English/Filipino content
- Store in organized structure

3. **Metadata Management**
- Track document sources
- Add politician tags
- Categorize by topic (laws, achievements, issues)
- Version control for updated documents

4. **Improve Chunking**
- Experiment with chunk sizes (500, 1000, 1500 tokens)
- Add overlap between chunks
- Preserve context for legal documents

### Phase 3: API & Interface Development (Week 5-6)

**Goals:**
- Build FastAPI backend
- Create production-ready Streamlit interface
- Add filtering and search features
- Implement source citation

**Tasks:**

1. **FastAPI Backend**
- Create `/query` endpoint
- Add `/ingest` endpoint for new documents
- Implement caching
- Add error handling

2. **Streamlit Frontend**
- Search interface with filters
- Results display with sources
- Politician comparison feature
- Timeline view for legislative changes

3. **Advanced Features**
- Filter by politician, date range, topic
- Compare multiple politicians
- Export results to PDF/Word
- Share results via link

### Phase 4: Optimization & Deployment (Week 7-8)

**Goals:**
- Optimize performance
- Deploy to production
- Set up monitoring
- Create documentation

**Tasks:**

1. **Performance Optimization**
- Optimize embedding generation
- Add result caching
- Improve query speed
- Reduce API costs

2. **Deployment**
- Containerize with Docker
- Deploy backend to Railway/Render
- Deploy frontend to Streamlit Cloud
- Set up CI/CD

3. **Monitoring & Maintenance**
- Set up logging
- Monitor API usage
- Track query performance
- Plan for updates

---

## Code Examples & Setup

### 1. Environment Setup

```bash
# requirements.txt for backend
langchain==0.1.0
chromadb==0.4.22
sentence-transformers==2.3.1
beautifulsoup4==4.12.3
requests==2.31.0
pypdf2==3.0.1
pdfplumber==0.10.3
fastapi==0.109.0
uvicorn==0.27.0
python-dotenv==1.0.0

# requirements.txt for frontend
streamlit==1.31.0
requests==2.31.0
pandas==2.2.0
plotly==5.18.0
```

### 2. Document Processing

```python
# backend/src/data_ingestion/processors/pdf_processor.py

import pdfplumber
from pathlib import Path

def extract_text_from_pdf(pdf_path: str) -> dict:
    """Extract text and metadata from PDF."""
    text_content = []
    metadata = {}
    
    with pdfplumber.open(pdf_path) as pdf:
        # Extract metadata
        metadata = {
            'filename': Path(pdf_path).name,
            'pages': len(pdf.pages),
            'author': pdf.metadata.get('Author', 'Unknown'),
            'creation_date': pdf.metadata.get('CreationDate', None)
        }
        
        # Extract text from all pages
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_content.append(text)
    
    return {
        'text': '\n\n'.join(text_content),
        'metadata': metadata
    }
```

### 3. Creating Embeddings

```python
# backend/src/embeddings/create_embeddings.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from pathlib import Path
import json

class DocumentEmbedder:
    def __init__(self, persist_directory: str = "./vector_db"):
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize vector store
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
    
    def add_document(self, text: str, metadata: dict):
        """Add a document to the vector store."""
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create documents with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy()
            doc_metadata['chunk_id'] = i
            documents.append({
                'text': chunk,
                'metadata': doc_metadata
            })
        
        # Add to vector store
        self.vectorstore.add_texts(
            texts=[doc['text'] for doc in documents],
            metadatas=[doc['metadata'] for doc in documents]
        )
        
        # Persist changes
        self.vectorstore.persist()
        
        return len(documents)

# Usage
embedder = DocumentEmbedder()
embedder.add_document(
    text="Republic Act No. 1234...",
    metadata={
        'source': 'Official Gazette',
        'type': 'law',
        'title': 'Republic Act No. 1234',
        'date': '2024-01-15'
    }
)
```

### 4. RAG Query System

```python
# backend/src/retrieval/rag_chain.py

from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate

class RAGQuerySystem:
    def __init__(self, persist_directory: str = "./vector_db"):
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load vector store
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        # Initialize LLM (using local Ollama)
        self.llm = Ollama(model="llama3.2:3b", temperature=0)
        
        # Create custom prompt
        template = """You are an expert on Philippine government and politics. 
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always cite the source of your information.

Context: {context}

Question: {question}

Answer:"""
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 5}  # Retrieve top 5 chunks
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
    
    def query(self, question: str) -> dict:
        """Query the system and return answer with sources."""
        result = self.qa_chain({"query": question})
        
        return {
            'answer': result['result'],
            'sources': [
                {
                    'content': doc.page_content[:200] + '...',
                    'metadata': doc.metadata
                }
                for doc in result['source_documents']
            ]
        }

# Usage
rag = RAGQuerySystem()
result = rag.query("What are Leni Robredo's major achievements?")
print(result['answer'])
for source in result['sources']:
    print(f"Source: {source['metadata']['title']}")
```

### 5. FastAPI Backend

```python
# backend/src/api/main.py

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
import sys
sys.path.append('..')
from retrieval.rag_chain import RAGQuerySystem

app = FastAPI(title="Philippine Politician RAG API")

# Initialize RAG system
rag = RAGQuerySystem()

class QueryRequest(BaseModel):
    question: str
    max_results: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document database."""
    try:
        result = rag.query(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# Run with: uvicorn main:app --reload
```

### 6. Streamlit Frontend

```python
# frontend/app.py

import streamlit as st
import requests
import json

# API configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Philippine Politician Summary Tool",
    page_icon="🇵🇭",
    layout="wide"
)

st.title("🇵🇭 Philippine Politician Summary Tool")
st.markdown("Search for information about Philippine politicians, laws, and government policies.")

# Sidebar filters
st.sidebar.header("Filters")
politician_filter = st.sidebar.text_input("Filter by politician name")
date_range = st.sidebar.date_input("Date range", [])

# Main search interface
query = st.text_input(
    "Ask a question:",
    placeholder="e.g., What are Leni Robredo's major achievements?"
)

if query:
    with st.spinner("Searching documents..."):
        try:
            # Call API
            response = requests.post(
                f"{API_URL}/query",
                json={"question": query, "max_results": 5}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Display answer
                st.subheader("Answer")
                st.write(data['answer'])
                
                # Display sources
                st.subheader("Sources")
                for i, source in enumerate(data['sources'], 1):
                    with st.expander(f"Source {i}: {source['metadata'].get('title', 'Unknown')}"):
                        st.write(f"**Type:** {source['metadata'].get('type', 'Unknown')}")
                        st.write(f"**Date:** {source['metadata'].get('date', 'Unknown')}")
                        st.write(f"**Source:** {source['metadata'].get('source', 'Unknown')}")
                        st.write("---")
                        st.write(source['content'])
            else:
                st.error(f"Error: {response.status_code}")
                
        except Exception as e:
            st.error(f"Error connecting to API: {str(e)}")

# Run with: streamlit run app.py
```

### 7. Docker Setup

```dockerfile
# backend/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY vector_db/ ./vector_db/

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml

version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend/vector_db:/app/vector_db
      - ./backend/data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
  
  frontend:
    build: ./frontend
    ports:
      - "8501:8501"
    depends_on:
      - backend
    environment:
      - API_URL=http://backend:8000

# Run with: docker-compose up
```

---

## Deployment Strategy

### Local Development
```bash
# Terminal 1: Run backend
cd backend
uvicorn src.api.main:app --reload

# Terminal 2: Run frontend
cd frontend
streamlit run app.py
```

### Production Deployment

#### Option 1: Railway (Easiest)
1. Push code to GitHub
2. Connect Railway to GitHub repo
3. Deploy backend and frontend as separate services
4. Set environment variables
5. **Cost:** Free tier available, ~$5-20/month for production

#### Option 2: Render
1. Create Render account
2. Deploy backend as Web Service
3. Deploy frontend as Web Service
4. Connect services
5. **Cost:** Free tier available, ~$7-25/month for production

#### Option 3: Fly.io
1. Install Fly CLI
2. `fly launch` in backend directory
3. `fly launch` in frontend directory
4. Configure scaling
5. **Cost:** Free tier available, pay as you go

#### Option 4: Self-Hosted VPS
1. Get VPS (DigitalOcean, Linode, Vultr)
2. Install Docker and Docker Compose
3. Deploy with `docker-compose up -d`
4. Set up nginx reverse proxy
5. **Cost:** $5-10/month for basic VPS

---

## Budget Estimates

### Development Phase (Months 1-2)

| Item | Cost |
|------|------|
| **LLM (Local Ollama)** | $0 |
| **Vector Database (ChromaDB)** | $0 |
| **Development Tools** | $0 |
| **Testing/Learning** | $0 |
| **Total** | **$0** |

### Production Phase (Monthly)

#### Minimal Budget Option
| Item | Cost |
|------|------|
| **LLM (Local Ollama)** | $0 |
| **Vector Database (ChromaDB self-hosted)** | $0 |
| **Hosting (Railway/Render free tier)** | $0 |
| **Domain (optional)** | $10/year |
| **Total** | **$0-1/month** |

#### Recommended Production Option
| Item | Cost |
|------|------|
| **LLM API (GPT-4o-mini or Claude Haiku)** | $10-30 |
| **Vector Database (Pinecone free tier)** | $0 |
| **Backend Hosting (Railway/Render)** | $7-15 |
| **Frontend Hosting (Streamlit Cloud)** | $0 |
| **Domain** | $1/month |
| **Total** | **$20-50/month** |

#### High-Performance Option
| Item | Cost |
|------|------|
| **LLM API (GPT-4 or Claude Sonnet)** | $50-100 |
| **Vector Database (Pinecone Pro)** | $70 |
| **Hosting (Dedicated VPS)** | $20-40 |
| **CDN & SSL** | $10 |
| **Monitoring** | $10 |
| **Total** | **$150-220/month** |

---

## Next Steps & Resources

### Immediate Next Steps

1. **Week 1: Environment Setup**
   - [ ] Set up Python virtual environment
   - [ ] Install Ollama and pull Llama 3.2 model
   - [ ] Install required Python packages
   - [ ] Create project folder structure

2. **Week 1-2: First Prototype**
   - [ ] Download 50 sample documents from Official Gazette
   - [ ] Process documents and create embeddings
   - [ ] Build basic RAG query system
   - [ ] Test with simple queries

3. **Week 3: Expand Data Collection**
   - [ ] Build web scraper for Official Gazette
   - [ ] Download 200+ documents
   - [ ] Organize by category (laws, politician profiles, news)
   - [ ] Add metadata tracking

4. **Week 4: Build Interface**
   - [ ] Create FastAPI backend
   - [ ] Build Streamlit frontend
   - [ ] Connect backend and frontend
   - [ ] Test end-to-end workflow

### Learning Resources

#### RAG & LangChain
- LangChain Documentation: https://python.langchain.com/docs/get_started/introduction
- RAG Tutorial: https://python.langchain.com/docs/use_cases/question_answering/
- ChromaDB Docs: https://docs.trychroma.com/

#### Philippine Government Data
- Official Gazette: https://www.officialgazette.gov.ph/
- Senate: https://www.senate.gov.ph/
- Congress: https://www.congress.gov.ph/
- Supreme Court: https://elibrary.judiciary.gov.ph/

#### Web Scraping
- BeautifulSoup Tutorial: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
- Scrapy: https://scrapy.org/
- Requests: https://requests.readthedocs.io/

#### FastAPI & Streamlit
- FastAPI: https://fastapi.tiangolo.com/
- Streamlit: https://docs.streamlit.io/

### Philippine Context Considerations

#### Language Handling
- Many documents mix English and Filipino (Tagalog)
- Consider using multilingual embeddings
- May need to handle code-switching in queries

#### Legal Terminology
- Philippine legal language has specific conventions
- Republic Acts (RA), Executive Orders (EO), etc.
- Consider creating a glossary for common terms

#### Data Quality
- Government websites may have inconsistent formatting
- PDFs vary widely in quality (scanned vs. digital)
- News sources may have bias - consider multiple sources

#### Privacy & Ethics
- Be careful with SALN data (financial disclosures)
- Verify information from multiple sources
- Add disclaimers about data accuracy
- Consider data retention policies

### Community & Support

#### Philippine Tech Communities
- Python PH: https://python.ph/
- DevCon Philippines: https://devcon.ph/
- Philippine Data Science Community

#### Getting Help
- LangChain Discord: https://discord.gg/langchain
- r/Philippines for context questions
- Stack Overflow for technical issues

---

## Appendix: Common Issues & Solutions

### Issue 1: Poor Retrieval Quality
**Symptoms:** RAG returns irrelevant documents

**Solutions:**
- Adjust chunk size (try 500, 1000, 1500)
- Increase chunk overlap (try 100-300 tokens)
- Experiment with different embedding models
- Add metadata filters to narrow search
- Improve document preprocessing

### Issue 2: Slow Query Performance
**Symptoms:** Queries take >5 seconds

**Solutions:**
- Reduce number of retrieved chunks (k=3 instead of k=5)
- Use smaller LLM model
- Implement caching for common queries
- Optimize vector database indices
- Use GPU for embeddings if available

### Issue 3: LLM Hallucinations
**Symptoms:** LLM makes up information not in documents

**Solutions:**
- Lower temperature (set to 0 for factual answers)
- Improve prompts (emphasize "only use provided context")
- Add citation requirements
- Implement fact-checking with multiple sources
- Consider using Claude (better at following instructions)

### Issue 4: Mixed Language Issues
**Symptoms:** Poor results for Filipino language queries

**Solutions:**
- Use multilingual embedding model
- Separate English and Filipino documents
- Translate queries to English before searching
- Use language-specific prompts
- Consider fine-tuning embeddings

### Issue 5: Out-of-Date Information
**Symptoms:** Returns old information when newer exists

**Solutions:**
- Add date-based ranking to retrieval
- Implement document versioning
- Regular data updates (weekly/monthly)
- Add "last updated" metadata to results
- Prioritize recent sources in ranking

---

## Contact & Feedback

This is a living document. As you progress through the project:
- Update timeline estimates based on actual progress
- Add new learnings and solutions
- Document challenges and workarounds
- Track costs and optimizations

Good luck with your project! 🇵🇭

---

**Document Version:** 1.0
**Created:** February 10, 2026
**Author:** Claude (with your project requirements)
