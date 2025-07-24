# Real Estate RAG Project

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain, Chroma vector store, and Groq/HuggingFace models. It demonstrates how to build a QA system that retrieves relevant documents and generates answers using LLMs.

## Features

- Loads and splits documents from URLs
- Embeds documents using HuggingFace models
- Stores embeddings in a persistent Chroma vector store
- Retrieves relevant documents for a given question
- Generates answers with sources using Groq LLM

## Project Structure

```
.
├── main.py
├── rag.py
├── requirements.txt
├── .env
├── resources/
│   └── vectorstore/
│       └── chroma.sqlite3
│       └── <embedding data>
```

## Usage

1. **Install dependencies**

   ```
   pip install -r requirements.txt
   ```

2. **Configure environment**  
   Add your API keys and settings to `.env`.

3. **Run the RAG pipeline**
   ```
   python rag.py
   ```

## Main Components

- [`rag.py`](rag.py): Core logic for loading data, embedding, storing, retrieving, and answering questions.
- [`resources/vectorstore/chroma.sqlite3`](resources/vectorstore/chroma.sqlite3): Persistent vector store for embeddings.

## Example

```python
answer, sources = generate_answer("Tell me the two teams playing today")
print(f"Answer: {answer}")
print(f"Sources: {sources}")
```
