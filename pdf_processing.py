import os
import hashlib
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List

import pymupdf as fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_index_id(pdf_paths: list) -> str:
    names = sorted([Path(p).name for p in pdf_paths])
    combined = "_".join(names)
    return hashlib.md5(combined.encode()).hexdigest()[:10]


def extract_text_from_pdf(path: str) -> List[Document]:
    docs = []
    with fitz.open(path) as pdf:
        for i in range(len(pdf)):
            page = pdf[i]
            text = str(page.get_text())
            meta = {"source": Path(path).name, "page": i + 1}
            docs.append(Document(page_content=text, metadata=meta))
    return docs


def split_text(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)


def load_and_index_pdfs(pdf_paths: list, index_base: str = "faiss_index") -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    index_id = get_index_id(pdf_paths)
    pdf_names = ", ".join([Path(p).name for p in pdf_paths])
    index_dir = os.path.join(index_base, index_id)

    if os.path.exists(index_dir):
        print(f"📂 Found existing index for [{pdf_names}] — loading from disk...")
        return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

    try:
        print(f"🔨 No index found for [{pdf_names}] — building...")
        all_docs = []
        for path in pdf_paths:
            page_docs = extract_text_from_pdf(str(path))
            chunks = split_text(page_docs)
            all_docs.extend(chunks)

        print(f"PDF Status: 🟢\nNo. of PDFs: {len(pdf_paths)} \nTotal no. of Chunks: {len(all_docs)}")

        vectorstore = FAISS.from_documents(all_docs, embeddings)
        os.makedirs(index_dir, exist_ok=True)
        vectorstore.save_local(index_dir)
        print(f"💾 Index saved → faiss_index/{index_id}/ [{pdf_names}]")
        return vectorstore

    except Exception as e:
        raise RuntimeError(f"[ERROR] PDF batch processing failed: {e}")