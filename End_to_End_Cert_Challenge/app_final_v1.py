import chainlit as cl
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_qdrant import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, VectorParams, Distance
from pathlib import Path
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# === CONFIGURATION ===
pdf_folder = "dataset"
collection_name = "reports_collection"
embedding_model = OpenAIEmbeddings()
embedding_dim = 1536
sentence_model_name = 'all-mpnet-base-v2'  # A good general-purpose sentence embedding model
sentence_model = SentenceTransformer(sentence_model_name)

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

# === Split PDFs into chunks ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# === Vector Store ===
chunk_texts = [chunk.page_content for chunk in chunks]
chunk_metadatas = [chunk.metadata for chunk in chunks]

vector_store = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embedding_model,
)
vector_store.add_texts(texts=chunk_texts, metadatas=chunk_metadatas)

# === Global state for session ===
selected_file = None

@cl.on_chat_start
async def start():
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    options = [{"label": pdf, "value": os.path.join(pdf_folder, pdf)} for pdf in pdf_files]
    options.sort(key=lambda x: x["label"])
    await cl.Message(content="👋 Welcome! Please type the name of the company's report you want to analyze.").send()
    report_list = "\n".join([f"📄 {opt['label']}" for opt in options])
    await cl.Message(content=f"Available reports:\n{report_list}").send()

@cl.on_message
async def respond(message: cl.Message):
    global selected_file

    if not selected_file:
        user_input = message.content.lower()
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        for pdf in pdf_files:
            if user_input in pdf.lower():
                selected_file = os.path.join(pdf_folder, pdf)
                company_name = Path(selected_file).stem.replace('-', ' ').title()
                await cl.Message(content=f"✅ You selected: {company_name}'s report. Ask your question!").send()
                return
        await cl.Message(content="❌ Report not found. Please try again.").send()
        return

    user_question = message.content
    qdrant_filter = Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value=selected_file)
            )
        ]
    )

    #results = vector_store.similarity_search(user_question, k=3, filter=qdrant_filter)
    search_query = f"{user_question} in {selected_file}"
    #results = vector_store.similarity_search(user_question, k=5, filter=qdrant_filter)
    results = vector_store.similarity_search(search_query, k=5)

    if not results:
        await cl.Message(content="😕 No relevant information found.").send()
        return

    relevant_sections = []
    question_embedding = sentence_model.encode(user_question)

    for doc in results:
        page_num = doc.metadata.get('page', '?')
        content = doc.page_content.strip()
        sentences = [s.strip() for s in content.split('. ') if s.strip()]

        if not sentences:
            continue

        sentence_embeddings = sentence_model.encode(sentences)

        # Calculate cosine similarity between question and sentence embeddings
        similarities = np.inner(question_embedding, sentence_embeddings)

        # Get the indices of the top 3 most similar sentences
        top_indices = np.argsort(similarities)[::-1][:3]

        if top_indices.size > 0:
            relevant_sentences = [sentences[i] for i in top_indices]
            relevant_sections.append(f"📄 Page {page_num}:\n" + ". ".join(relevant_sentences) + ".")

    if relevant_sections:
        response = "\n\n---\n\n".join(relevant_sections)[:1500]
    else:
        response = "😕 I couldn't find specific relevant sentences in the context."

    await cl.Message(content=response).send()