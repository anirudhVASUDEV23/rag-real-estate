import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from uuid import uuid4
from dotenv import load_dotenv # REMOVED/COMMENTED OUT
from pathlib import Path
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

import os # Keep this for the debug print, can be removed later if desired

load_dotenv()


# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None

# IMPORTANT: Replace this with your actual Groq API Key
# WARNING: Hardcoding API keys is generally NOT recommended for production.
# This is a temporary workaround because Streamlit Cloud secrets are not being
# injected into environment variables for your app.


def initialize_components():
    global llm, vector_store

    if llm is None:
        # Debug print to verify environment variable (will still be None, but shows the attempt)
        print(f"DEBUG: GROQ_API_KEY from os.environ.get: {os.environ.get('GROQ_API_KEY')}")
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.9,
            max_tokens=500,
          
        )

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )


def process_urls(urls):
    """
    This function scraps data from a url and stores it in a vector db
    :param urls: input urls
    :return:
    """
    yield "Initializing components"
    initialize_components()

    yield "Resetting vector db"
    vector_store.reset_collection()

    yield "Loading data"
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    print(f"DEBUG: Length of loaded data documents: {len(data)}") # DEBUG PRINT

    yield "Splitting data"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    docs = text_splitter.split_documents(data)
    print(f"DEBUG: Number of documents after splitting: {len(docs)}") # DEBUG PRINT

    yield "Adding data to vector db"
    uuids = [str(uuid4()) for _ in range(len(docs))]

    # IMPORTANT: Check if docs list is empty before adding to vector store
    if not docs:
        print("WARNING: No documents extracted from URLs. Skipping addition to vector database.")
        yield "No content extracted from URLs. Please check URLs or content."
        return # Exit the function, as there's nothing to add

    vector_store.add_documents(docs, ids=uuids)

    yield "Done now ask your questions! ✅✅"


def generate_answer(question):
    if not vector_store:
        raise RuntimeError("Vector store is not initialized. Please call process_urls() first.")
    chain=RetrievalQAWithSourcesChain.from_llm(llm,retriever=vector_store.as_retriever())
    result=chain.invoke({"question": question},return_only_outputs=True)
    sources=result.get("sources","")

    return result["answer"],sources


# if __name__ == "__main__":
#     urls = [
#         "https://www.cricbuzz.com/live-cricket-scores/105778/ind-vs-eng-4th-test-india-tour-of-england-2025",
#         "https://indianexpress.com/section/sports/cricket/live-score/england-vs-india-4th-test-live-score-full-scorecard-highlights-anderson-tendulkar-trophy-2025-enin07232025250828/" # Corrected URL
#     ]

#     # This part assumes you're running it outside a Streamlit UI (e.g., in a script)
#     # If this is part of your Streamlit app's main file, this block might be handled differently
#     # in the Streamlit app's control flow (e.g., triggered by a button click).
#     # For now, this will execute process_urls immediately on app startup in Streamlit.
#     for status in process_urls(urls):
#         print(status)

#     answer, sources = generate_answer("Tell me the two teams playing today")
#     print(f"Answer: {answer}")
#     print(f"Sources: {sources}")