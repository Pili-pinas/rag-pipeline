"""Ingest documents from data/raw/ into Pinecone."""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
METADATA_FILE = Path(__file__).parent.parent / "data" / "metadata.json"


def load_metadata() -> dict[str, dict]:
    """Load metadata index keyed by filename."""
    if not METADATA_FILE.exists():
        print("No metadata.json found. Run download_samples.py first.")
        sys.exit(1)

    with open(METADATA_FILE) as f:
        entries = json.load(f)

    return {entry["filename"]: entry for entry in entries}


def main():
    # Verify env vars
    for var in ("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX"):
        if not os.environ.get(var):
            print(f"Missing environment variable: {var}")
            print("Copy .env.example to .env and fill in your keys.")
            sys.exit(1)

    metadata_index = load_metadata()

    # Collect documents
    txt_files = sorted(DATA_DIR.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {DATA_DIR}")
        print("Run download_samples.py first.")
        sys.exit(1)

    print(f"Found {len(txt_files)} documents to ingest.\n")

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    all_texts = []
    all_metadatas = []

    for filepath in txt_files:
        content = filepath.read_text(encoding="utf-8")
        filename = filepath.name
        meta = metadata_index.get(filename, {"title": filename, "source": "Unknown", "type": "Unknown", "date": "Unknown"})

        chunks = splitter.split_text(content)
        print(f"  {meta.get('title', filename)}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_metadatas.append(
                {
                    "title": meta.get("title", filename),
                    "source": meta.get("source", "Unknown"),
                    "type": meta.get("type", "Unknown"),
                    "date": meta.get("date", "Unknown"),
                    "chunk_index": i,
                    "filename": filename,
                }
            )

    print(f"\nTotal chunks: {len(all_texts)}")
    print("Generating embeddings and upserting to Pinecone...")

    # Create embeddings and upsert
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

    vectorstore = PineconeVectorStore.from_texts(
        texts=all_texts,
        metadatas=all_metadatas,
        embedding=embeddings,
        index_name=os.environ["PINECONE_INDEX"],
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
    )

    print(f"\nDone! Ingested {len(all_texts)} chunks into Pinecone index '{os.environ['PINECONE_INDEX']}'.")


if __name__ == "__main__":
    main()
