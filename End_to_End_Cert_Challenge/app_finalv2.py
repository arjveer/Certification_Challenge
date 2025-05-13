import chainlit as cl
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_qdrant import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, VectorParams, Distance
from pathlib import Path
import os
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI  # Or any other LLM you prefer

# === CONFIGURATION ===
pdf_folder = "dataset"
collection_name = "reports_collection"
#embedding_model = OpenAIEmbeddings()
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
embedding_dim = 1536
llm = OpenAI()  # Initialize your language model

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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
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

# === RetrievalQA Chain ===
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Or "map_reduce", "refine", "map_rerank" depending on your needs
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=False  # Set to False to not return context
)

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
    search_query = f"{user_question} in {selected_file}"
    #results = vector_store.similarity_search(user_question, k=5, filter=qdrant_filter)
    #results = vector_store.similarity_search(search_query, k=3)
    # Use the RetrievalQA chain for answering
    response = qa.run(query=search_query)
    await cl.Message(content=response).send()