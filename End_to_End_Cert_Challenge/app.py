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
    doc.metadata["source"] = str(source_path.as_posix())

# === Split PDFs into chunks ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# === Vector Store ===
#embeddings = embedding_model.embed_documents([doc.page_content for doc in docs])
#metadatas = [doc.metadata for doc in docs]

#embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])
#metadatas = [chunk.metadata for chunk in chunks]

chunk_texts = [chunk.page_content for chunk in chunks]
chunk_metadatas = [chunk.metadata for chunk in chunks]

vector_store = Qdrant(
    #documents=docs,
    client=client,
    collection_name=collection_name,
    embeddings=embedding_model,
)
vector_store.add_texts(texts=chunk_texts, metadatas=chunk_metadatas)

# === Global state for session ===
selected_file = None


@cl.on_chat_start
async def start():
    doc_sources = sorted(set(d.metadata["source"] for d in docs))
    options = [{"label": Path(src).name, "value": src} for src in doc_sources]

    await cl.Message(content="📄 Select the PDF you want to query:").send()
    await cl.ChatPrompt(
        id="file_selector",
        name="Select File",
        description="Choose which PDF to query",
        type="select",
        options=options
    ).send()


@cl.on_message
async def respond(message: cl.Message):
    global selected_file
    
    # If no file is selected, treat the message as a file selection
    if not selected_file:
        selected_file = message.content
        await cl.Message(content=f"✅ You selected: `{Path(selected_file).name}`. Ask your question!").send()
        return

    # If a file is already selected, process the question
    user_question = message.content
    qdrant_filter = Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value=selected_file)
            )
        ]
    )

    results = vector_store.similarity_search(user_question, k=5, filter=qdrant_filter)
    if not results:
        response = "😕 No relevant content found for your query."
    else:
        chunks = [
            f"{doc.page_content}\n--- Page {doc.metadata.get('page', '?')}"
            for doc in results
        ]
        response = "\n\n".join(chunks)

    await cl.Message(content=response).send()
