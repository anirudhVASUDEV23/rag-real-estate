

# from uuid import uuid4
# from dotenv import load_dotenv # REMOVED/COMMENTED OUT
# from pathlib import Path
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain_community.document_loaders import UnstructuredURLLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_groq import ChatGroq
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# import os # Keep this for the debug print, can be removed later if desired

# load_dotenv()


# # Constants
# CHUNK_SIZE = 1000
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
# COLLECTION_NAME = "real_estate"

# llm = None
# vector_store = None


# # IMPORTANT: Replace this with your actual Groq API Key
# # WARNING: Hardcoding API keys is generally NOT recommended for production.
# # This is a temporary workaround because Streamlit Cloud secrets are not being
# # injected into environment variables for your app.


# def initialize_components():
#     global llm, vector_store

#     if llm is None:
#         # Debug print to verify environment variable (will still be None, but shows the attempt)
#         print(f"DEBUG: GROQ_API_KEY from os.environ.get: {os.environ.get('GROQ_API_KEY')}")
#         llm = ChatGroq(
#             model="llama-3.3-70b-versatile",
#             temperature=0.9,
#             max_tokens=500,
           
#         )

#     if vector_store is None:
#         ef = HuggingFaceEmbeddings(
#             model_name=EMBEDDING_MODEL,
#             model_kwargs={"trust_remote_code": True}
#         )
#         VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
#         vector_store = Chroma(
#             collection_name=COLLECTION_NAME,
#             embedding_function=ef,
#             persist_directory=str(VECTORSTORE_DIR)
#         )


# def process_urls(urls):
#     """
#     This function scraps data from a url and stores it in a vector db
#     :param urls: input urls
#     :return:
#     """
#     yield "Initializing components"
#     initialize_components()

#     yield "Resetting vector db"
#     vector_store.reset_collection()

#     yield "Loading data"
#     loader = UnstructuredURLLoader(urls=urls)
#     data = loader.load()
#     print(f"DEBUG: Length of loaded data documents: {len(data)}") # DEBUG PRINT

#     yield "Splitting data"
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=["\n\n", "\n", ".", " "],
#         chunk_size=CHUNK_SIZE
#     )
#     docs = text_splitter.split_documents(data)
#     print(f"DEBUG: Number of documents after splitting: {len(docs)}") # DEBUG PRINT

#     yield "Adding data to vector db"
#     uuids = [str(uuid4()) for _ in range(len(docs))]

#     # IMPORTANT: Check if docs list is empty before adding to vector store
#     if not docs:
#         print("WARNING: No documents extracted from URLs. Skipping addition to vector database.")
#         yield "No content extracted from URLs. Please check URLs or content."
#         return # Exit the function, as there's nothing to add

#     vector_store.add_documents(docs, ids=uuids)

#     yield "Done now ask your questions! ✅✅"


# def generate_answer(question):
#     if not vector_store:
#         raise RuntimeError("Vector store is not initialized. Please call process_urls() first.")
#     chain=RetrievalQAWithSourcesChain.from_llm(llm,retriever=vector_store.as_retriever())
#     result=chain.invoke({"question": question},return_only_outputs=True)
#     sources=result.get("sources","")

#     return result["answer"],sources


# # if __name__ == "__main__":
# #     urls = [
# #         "https://www.cricbuzz.com/live-cricket-scores/105778/ind-vs-eng-4th-test-india-tour-of-england-2025",
# #         "https://indianexpress.com/section/sports/cricket/live-score/england-vs-india-4th-test-live-score-full-scorecard-highlights-anderson-tendulkar-trophy-2025-enin07232025250828/" # Corrected URL
# #     ]

# #     # This part assumes you're running it outside a Streamlit UI (e.g., in a script)
# #     # If this is part of your Streamlit app's main file, this block might be handled differently
# #     # in the Streamlit app's control flow (e.g., triggered by a button click).
# #     # For now, this will execute process_urls immediately on app startup in Streamlit.
# #     for status in process_urls(urls):
# #         print(status)

# #     answer, sources = generate_answer("Tell me the two teams playing today")
# #     print(f"Answer: {answer}")
# #     print(f"Sources: {sources}")












import sys
# This line is crucial for ChromaDB compatibility on Streamlit Cloud/some Linux environments
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain

# We are switching from Groq to OpenAI due to API key issues
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Tuple
from langchain_core.documents import Document # Import Document class for static text

# --- Constants ---
CHROMA_DB_DIR = "chroma_db"
LLM_MODEL_NAME = "gpt-3.5-turbo" # Using a reliable OpenAI model
LLM_TEMPERATURE = 0.7 # A balanced temperature for creative but factual responses
LLM_MAX_TOKENS = 500 # Max tokens for the LLM's response

# --- Global Variables ---
llm = None
vector_store = None

# --- Function to Initialize LLM and Vector Store ---
def initialize_components():
    global llm, vector_store

    # Initialize LLM if not already done
    if llm is None:
        # Debugging: Print to check if API key is loaded from environment
        print(f"DEBUG: OPENAI_API_KEY from os.environ.get: {os.environ.get('OPENAI_API_KEY')}")
        openai_api_key = os.environ.get('OPENAI_API_KEY')

        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in Streamlit Secrets.")

        llm = ChatOpenAI(
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            api_key=openai_api_key
        )
        print("LLM (ChatOpenAI) initialized successfully.")

    # Initialize Vector Store if not already done
    if vector_store is None:
        # Embeddings also use the OpenAI API key
        embeddings = OpenAIEmbeddings(api_key=os.environ.get('OPENAI_API_KEY'))

        try:
            # Attempt to load existing vector store
            if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
                print(f"Loading existing vector store from {CHROMA_DB_DIR}")
                vector_store = Chroma(
                    persist_directory=CHROMA_DB_DIR,
                    embedding_function=embeddings
                )
            else:
                # If directory doesn't exist or is empty, create new
                print(f"Vector store directory {CHROMA_DB_DIR} not found or empty, creating new one.")
                if os.path.exists(CHROMA_DB_DIR): # Ensure it's clean if it existed but was empty
                    import shutil
                    shutil.rmtree(CHROMA_DB_DIR)
                vector_store = Chroma(
                    persist_directory=CHROMA_DB_DIR,
                    embedding_function=embeddings
                )

        except Exception as e:
            # Fallback for any other loading errors, create new
            print(f"Error loading vector store: {e}. Creating a new one.")
            if os.path.exists(CHROMA_DB_DIR):
                import shutil
                shutil.rmtree(CHROMA_DB_DIR)
            vector_store = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings
            )
        print("Vector store initialized successfully.")
    print("All components initialized! ✅")


# --- Function to Process URLs and Populate DB ---
def process_urls(urls: List[str]) -> None:
    global vector_store
    if vector_store is None:
        initialize_components() # Ensure components are initialized before use

    print(f"Attempting to load data from URLs: {urls}")

    # --- START OF STATIC TEXT WORKAROUND BLOCK ---
    # This block temporarily bypasses URL loading to verify the rest of the RAG pipeline.
    # We will remove or modify this later once we've confirmed the core RAG components work.
    
    # Example static text to use for testing
    static_text_content = [
        "Hyderabad is the capital and largest city of the Indian state of Telangana. It is a major center for the technology industry.",
        "The city is known for its rich history, culture, and architecture, including the Charminar and Golconda Fort. It was founded in 1591 by Muhammad Quli Qutb Shah.",
        "Hyderabad is also famous for its delicious cuisine, especially Hyderabadi Biryani. It is often referred to as the 'City of Pearls'.",
        "The IT hub, HITEC City, is a prominent technology township in Hyderabad, home to many multinational corporations."
    ]

    # Instead of loading from URLs, create documents directly from static text
    docs = [Document(page_content=text) for text in static_text_content]
    print(f"DEBUG: Using static text documents for testing. Length: {len(docs)}")
    # --- END OF STATIC TEXT WORKAROUND BLOCK ---

    # --- The original loader.load() block below is now commented out/replaced by the static text above ---
    # try:
    #     loader = UnstructuredURLLoader(urls=urls)
    #     docs = loader.load()
    #     print(f"DEBUG: Length of loaded data documents: {len(docs)}")
    # except Exception as e:
    #     print(f"Error fetching or processing URLs: {e}")
    #     return # Crucial: if error, return None


    if not docs:
        print("WARNING: No documents extracted (even from static text). Skipping addition to vector database.")
        return

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(docs)
    print(f"DEBUG: Number of documents after splitting: {len(texts)}")

    # Add to vector database
    if texts:
        embeddings = OpenAIEmbeddings(api_key=os.environ.get('OPENAI_API_KEY'))
        if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
            print(f"Adding {len(texts)} new documents to existing vector store.")
            vector_store.add_documents(texts)
        else:
            print(f"Creating new vector store with {len(texts)} documents.")
            vector_store = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=CHROMA_DB_DIR
            )
        vector_store.persist() # Save changes to disk
        print(f"Successfully added {len(texts)} documents to vector database.")
    else:
        print("WARNING: No texts generated after splitting documents. Vector database remains unchanged.")

    # Removed the outer try-except block here as the URL loading is now bypassed by static text.
    # Any exceptions in splitting/adding will still be caught, but not from URL fetching.


# --- Function to Generate Answer ---
def generate_answer(question: str) -> Tuple[str, str]:
    global llm, vector_store
    if llm is None or vector_store is None:
        initialize_components() # Ensure components are initialized

    # Debugging: Check if vector store has any documents before querying
    if vector_store._collection.count() == 0:
        print("DEBUG: Vector store is empty. LLM will answer from general knowledge.")
        # If the DB is empty, we can still try to get a general answer from the LLM
        # This will bypass retrieval, just calling the LLM directly
        try:
            # A simple prompt for general knowledge without context
            direct_response = llm.invoke(question)
            return direct_response.content, "No sources found (database empty)."
        except Exception as e:
            return f"Error generating direct answer from LLM: {e}", "No sources found."

    # If vector store has documents, proceed with RAG chain
    # Setting up the RAG chain
    chain = RetrievalQAWithSourcesChain.from_llm(llm, retriever=vector_store.as_retriever())

    print(f"Generating answer for question: {question}")
    try:
        result = chain.invoke({"question": question}, return_only_outputs=True)
        answer = result.get("answer", "No answer generated.")
        sources = result.get("sources", "No sources found.")
        print("Answer generated successfully.")
        return answer, sources
    except Exception as e:
        print(f"Error generating answer with RAG chain: {e}")
        return f"An error occurred: {e}", "No sources found due to error."


# --- Local Testing Block (Keep commented out for Streamlit Cloud deployment) ---
if __name__ == "__main__":
    # This block is for local testing purposes.
    # It should typically be commented out when deploying to Streamlit Cloud,
    # as Streamlit's main.py will handle the app's execution flow.

    print("--- Running rag.py in local test mode ---")
    initialize_components()

    # Example URLs for local testing (might still fail on cloud deployment)
    test_urls = [
        "https://en.wikipedia.org/wiki/Hyderabad"
    ]

    print("\n--- Processing URLs ---")
    process_urls(test_urls) # This will still use the static text if the block is active

    print("\n--- Asking a question ---")
    question = "What is Hyderabad known for?"
    answer, sources = generate_answer(question)
    print("\nAnswer:")
    print(answer)
    print("\nSources:")
    print(sources)

    print("\n--- Asking another question ---")
    question = "What is the history of Hyderabad?"
    answer, sources = generate_answer(question)
    print("\nAnswer:")
    print(answer)
    print("\nSources:")
    print(sources)