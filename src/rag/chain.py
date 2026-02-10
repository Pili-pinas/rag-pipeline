"""RAG query engine using Pinecone + OpenAI."""

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.rag.prompts import SYSTEM_PROMPT, QA_PROMPT_TEMPLATE


def _format_docs(docs: list) -> str:
    """Format retrieved documents into a single context string."""
    parts = []
    for doc in docs:
        meta = doc.metadata
        header = f"[{meta.get('title', 'Unknown')} | {meta.get('date', 'Unknown')}]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class RAGQueryEngine:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )

        self.vectorstore = PineconeVectorStore(
            index_name=os.environ["PINECONE_INDEX"],
            embedding=self.embeddings,
            pinecone_api_key=os.environ["PINECONE_API_KEY"],
        )

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", QA_PROMPT_TEMPLATE),
            ]
        )

    def query(self, question: str, max_results: int = 5) -> dict:
        """Query the RAG system and return an answer with sources."""
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": max_results}
        )

        # Retrieve relevant docs
        docs = retriever.invoke(question)

        # Build the chain
        chain = (
            {"context": lambda _: _format_docs(docs), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke(question)

        sources = [
            {
                "content": doc.page_content[:300],
                "title": doc.metadata.get("title", "Unknown"),
                "source": doc.metadata.get("source", "Unknown"),
                "type": doc.metadata.get("type", "Unknown"),
                "date": doc.metadata.get("date", "Unknown"),
            }
            for doc in docs
        ]

        return {"answer": answer, "sources": sources}
