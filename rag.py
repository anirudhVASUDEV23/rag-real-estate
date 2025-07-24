# import sys
# __import__('pysqlite3')
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

##load_dotenv()


# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)

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

    yield "Splitting data"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    docs = text_splitter.split_documents(data)

    yield "Adding data to vector db"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield "Done now ask your questions! ✅✅"


def generate_answer(question):
    if not vector_store:
        raise RuntimeError("Vector store is not initialized. Please call process_urls() first.")
    chain=RetrievalQAWithSourcesChain.from_llm(llm,retriever=vector_store.as_retriever())
    result=chain.invoke({"question": question},return_only_outputs=True)
    sources=result.get("sources","")

    return result["answer"],sources 


if __name__ == "__main__":
    urls = [
        "https://www.cricbuzz.com/live-cricket-scores/105778/ind-vs-eng-4th-test-india-tour-of-england-2025",
        "https://indianexpress.com/section/sports/cricket/live-score/england-vs-india-4th-test-live-score-full-scorecard-highlights-anderson-tendulkar-trophy-2025-enin07232025250828/"
    ]

    process_urls(urls)

    answer, sources = generate_answer("Tell me the two teams playing today")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")
