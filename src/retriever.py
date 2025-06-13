# src/retriever.py
"""
Hybrid Retriever
────────────────
• Dense vector search  (Gemini Embeddings + FAISS)
• Lexical BM25 search  (token-match)
Returns an EnsembleRetriever that mixes both scores.

Chunk size = 500 tokens, overlap = 200 – good for legal clauses.
"""

from pathlib import Path
from typing import List
import os
import src.logger as log

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document

lg = log.get("retriever")

# ──────────────────────────────────────────────────────────────────────────
# Build hybrid retriever
# ──────────────────────────────────────────────────────────────────────────
def build_index(
    pdf_path: str,
    api_key: str,
    k: int = 8,
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
):
    """
    Returns an EnsembleRetriever that combines dense + BM25.
    - k : top-k docs per retriever
    - dense_weight / bm25_weight : scoring weights
    """
    if not Path(pdf_path).exists():
        lg.error(f"PDF not found: {pdf_path}")
        raise FileNotFoundError(pdf_path)

    lg.info("Loading PDF …")
    pages = PyPDFLoader(pdf_path).load()          # List[Document]

    lg.info("Splitting into chunks … (500/200)")
    splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    lg.info(f"Loaded {len(pages)} pages → {len(chunks)} chunks")

    # Dense embeddings + FAISS
    lg.info("Building dense FAISS index …")
    emb = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )
    store = FAISS.from_documents(chunks, emb)
    dense_ret = store.as_retriever(search_kwargs={"k": k})

    # BM25 lexical retriever
    lg.info("Building BM25 index …")
    bm25_ret = BM25Retriever.from_documents(chunks)
    bm25_ret.k = k

    # Ensemble (hybrid) retriever
    hybrid = EnsembleRetriever(
        retrievers=[dense_ret, bm25_ret],
        weights=[dense_weight, bm25_weight],
    )
    lg.info("Hybrid retriever ready ✅")
    return hybrid
