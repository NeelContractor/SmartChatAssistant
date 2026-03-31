# SmartChat Assistant

A powerful AI chat application built with **Streamlit + LangGraph + Ollama** that supports:

* Multi-threaded conversations
* PDF-based Retrieval-Augmented Generation (RAG)
* Web search
* Stock price lookup
* Calculator tool
* Persistent memory using SQLite checkpoints

---

## Features

### 1. Multi-Threaded Chat

* Each conversation has a unique `thread_id`
* Threads are stored and can be reopened or deleted
* Chat history persists using LangGraph checkpoints

### 2. PDF RAG (Retrieval-Augmented Generation)

* Upload a PDF per thread
* Automatically:

  * Splits into chunks
  * Creates embeddings
  * Stores in FAISS vector DB
* Queries retrieve relevant chunks for accurate answers

### 3. Smart Context Injection (Important)

* Even if the LLM **fails to call tools**, document context is:

  * Retrieved manually
  * Injected into the system prompt
* Ensures reliable answers even with small models

### 4. Tooling Support

The assistant can use:

* `rag_tool` → query uploaded PDFs
* `DuckDuckGoSearchRun` → web search
* `calculator` → math operations
* `get_stock_price` → stock API (Alpha Vantage)

### 5. Streaming Responses

* Uses LangGraph streaming
* Displays tool usage in real-time

### 6. Persistent Memory (SQLite)

* Uses `SqliteSaver`
* Stores:

  * Messages
  * Tool calls
  * State transitions

---

## Architecture

### Frontend (Streamlit)

Handles:

* UI rendering
* Thread management
* File uploads
* Chat display
* Streaming responses

### Backend (LangGraph Agent)

Handles:

* LLM reasoning
* Tool calling
* RAG retrieval
* State persistence

### Flow

```text
User Input
   ↓
chat_node (LLM)
   ↓
[optional] Tool Call
   ↓
tools node executes
   ↓
chat_node (final response)
   ↓
Stream to UI
```

---

## 📂 Project Structure

```
.
├── app.py              # Streamlit frontend
├── backend.py          # LangGraph agent + tools + RAG
├── chatbot.db          # SQLite checkpoint DB
├── .env                # Environment variables
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repo

```bash
git clone <your-repo-url>
cd <repo>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install & Run Ollama

Download Ollama:
[https://ollama.com](https://ollama.com)

Pull a model:

```bash
ollama pull qwen3:4b
```

> ⚠️ Recommended: use **qwen3:4b or higher** (0.6b is weak for tool calling)

### 4. Environment Variables

Create `.env`:

```env
OLLAMA_MODEL=qwen3:0.6b
```

### 5. Run the App

```bash
streamlit run app.py
```

---

## How RAG Works

### Step 1: Upload PDF

* Stored temporarily
* Loaded using `PyPDFLoader`

### Step 2: Chunking

* `RecursiveCharacterTextSplitter`
* Chunk size: 1000
* Overlap: 200

### Step 3: Embeddings

* Model: `nomic-embed-text`

### Step 4: Vector Store

* FAISS in-memory store

### Step 5: Retrieval

* Top-K similarity search (k=4)

### Step 6: Injection

* Retrieved chunks inserted into system prompt

---

## Tools Overview

### 1. rag_tool

```python
rag_tool(query: str)
```

* Retrieves relevant document chunks
* Returns context + metadata

### 2. calculator

```python
calculator(first_num, second_num, operation)
```

* add, sub, mul, div

### 3. get_stock_price

```python
get_stock_price(symbol)
```

* Uses Alpha Vantage API

### 4. search_tool

* DuckDuckGo web search

---

## Thread & Memory System

* Each thread = unique UUID
* Stored in:

  * `checkpoints` table
  * `writes` table
* Managed via `SqliteSaver`

### Delete Thread

* Removes:

  * UI state
  * Database checkpoints

---

## Future Improvements

* Persist vector DB (Chroma / Pinecone)
* Support multiple PDFs per thread
* Better tool selection routing
