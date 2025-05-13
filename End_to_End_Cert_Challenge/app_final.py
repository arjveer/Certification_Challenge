import chainlit as cl
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_qdrant import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, VectorParams, Distance
from pathlib import Path
import os

# === CONFIGURATION ===
pdf_folder = "dataset"
collection_name = "reports_collection"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
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
    # Get list of PDF files from dataset folder
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    
    # Create a more user-friendly display of options
    options = []
    for pdf in pdf_files:
        # Remove .pdf extension and convert to title case for better display
        #display_name = pdf.replace('.pdf', '').replace('-', ' ').title()
        options.append({
            "label": pdf,
            "value": os.path.join(pdf_folder, pdf)
        })

    # Sort options alphabetically
    options.sort(key=lambda x: x["label"])

    # Send welcome message
    await cl.Message(
        content="👋 Welcome! I can help you analyze 10-K reports. Please select a company's report to begin:"
    ).send()

    # Create a list of available reports
    report_list = "\n".join([f"📄 {opt['label']}" for opt in options])
    await cl.Message(
        content=f"Available reports:\n{report_list}\n\nPlease type the name of the company you want to analyze."
    ).send()

@cl.on_message
async def respond(message: cl.Message):
    global selected_file
    
    # If no file is selected, treat the message as a file selection
    if not selected_file:
        # Find the matching file based on user input
        user_input = message.content.lower()
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        
        for pdf in pdf_files:
            if user_input in pdf.lower():
                selected_file = os.path.join(pdf_folder, pdf)
                print(selected_file)
                company_name = Path(selected_file).stem.replace('-', ' ').title()
                await cl.Message(
                    content=f"✅ You selected: {company_name}'s 10-K report. What would you like to know about it?"
                ).send()
                return
        
        # If no match found
        await cl.Message(
            content="❌ I couldn't find that company's report. Please try again with one of the available companies."
        ).send()
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
    search_query = f"{user_question} in {selected_file}"
    #results = vector_store.similarity_search(user_question, k=5, filter=qdrant_filter)
    results = vector_store.similarity_search(search_query, k=5)
    if not results:
        response = "😕 I couldn't find any relevant information in the report for your query. Please try rephrasing your question."
    else:
        chunks = []
        relevant_sections = []
        for doc in results:
            page_num = doc.metadata.get('page', '?')
            content = doc.page_content.strip()
            #chunks.append(f"📄 Page {page_num}:\n{content}")

        # Simple approach: Look for sentences containing keywords from the question
        keywords = user_question.lower().split()
        relevant_sentences = [
            s for s in content.split('. ') if any(keyword in s.lower() for keyword in keywords)
        ]

        if relevant_sentences:
            relevant_sections.append(f"📄 Page {page_num}:\n" + ". ".join(relevant_sentences) + ".")
        else:
            # If no specific sentences found, include a snippet from the beginning
            snippet = content[:200] + "..." if len(content) > 200 else content
            relevant_sections.append(f"📄 Page {page_num}:\n{snippet}")

    response = "\n\n---\n\n".join(relevant_sections)[:1500]  # Limit overall response length

        
    #response = "\n\n---\n\n".join(chunks)

    await cl.Message(content=response).send()