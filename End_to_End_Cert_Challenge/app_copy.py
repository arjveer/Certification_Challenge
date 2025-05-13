import chainlit as cl
from langchain_community.document_loaders import PyPDFDirectoryLoader
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === CONFIGURATION ===
pdf_folder = "dataset"
collection_name = "reports_collection"
embedding_dim = 1536

# === Load and Normalize PDFs ===
loader = PyPDFDirectoryLoader(pdf_folder)
docs = loader.load()
for doc in docs:
    source_path = Path(doc.metadata.get("source", ""))
    doc.metadata["source"] = str(source_path.as_posix())

# === Split Documents into Chunks ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# === Global state for session ===
selected_file = None

@cl.on_chat_start
async def start():
    await cl.Message(content="Hello! Please select a file to begin.").send()

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
    response = "Querying is temporarily disabled."
    await cl.Message(content=response).send()