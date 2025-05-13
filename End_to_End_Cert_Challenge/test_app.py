import chainlit as cl
from langchain_community.document_loaders import PyPDFDirectoryLoader
#from langchain_community.vectorstores import Qdrant
from langchain_qdrant import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, VectorParams, Distance
from pathlib import Path
import os



# === CONFIGURATION ===
pdf_folder = "dataset"
collection_name = "reports_collection"
embedding_model = OpenAIEmbeddings()
embedding_dim = 1536




# === INIT Qdrant ===
client = QdrantClient(":memory:")
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
)

# === Load and Normalize PDFs ===
loader = PyPDFDirectoryLoader(pdf_folder)
docs = loader.load()
for doc in docs:
    source_path = Path(doc.metadata.get("source", ""))
    doc.metadata["source"] = str(source_path.as_posix()).lower()

print(f"Selected File: {doc}")
for i, doc in enumerate(docs[:5]):  # Print metadata of the first 5 docs
    print(f"Document {i} Source Metadata: {doc.metadata.get('source')}")