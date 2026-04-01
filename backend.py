from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from duckduckgo_search import DDGS
from langchain_core.tools import tool as lc_tool
from tavily import TavilyClient
import requests
import psycopg

load_dotenv()

# ---------------------------------------------------------------------------
# LLM + Embeddings
# ---------------------------------------------------------------------------
LLM_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")

llm = ChatOllama(model=LLM_MODEL, base_url="http://localhost:11434")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ---------------------------------------------------------------------------
# PDF retriever store (per thread)
# ---------------------------------------------------------------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """Build a FAISS retriever for the uploaded PDF and store it for the thread."""
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Static tools
# ---------------------------------------------------------------------------

@lc_tool
def search_tool(query: str) -> str:
    """
    Search the web for current information. Use this for any question about
    recent events, news, prices, people, weather, or anything needing up-to-date info.
    Always call this tool when the user asks about current or recent information.
    """
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=True,   
        )

        # Direct answer if available
        output = []
        if response.get("answer"):
            output.append(f"**Summary:** {response['answer']}\n")

        for r in response.get("results", []):
            title = r.get("title", "No title")
            content = r.get("content", "No summary")
            url = r.get("url", "")
            output.append(f"**{title}**\n{content}\nSource: {url}")

        return "\n\n---\n\n".join(output) if output else f"No results found for: {query}"

    except Exception as e:
        return f"Search error: {type(e).__name__}: {e}"


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage.
    """
    api_key = os.getenv("API_KEY")
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    )
    r = requests.get(url)
    return r.json()


# ---------------------------------------------------------------------------
# Dynamic RAG tool factory
# thread_id baked into a closure — LLM only needs to call rag_tool(query).
# ---------------------------------------------------------------------------

def make_rag_tool(thread_id: str):
    @tool
    def rag_tool(query: str) -> dict:
        """
        Retrieve relevant information from the uploaded PDF for this chat.
        Call this whenever the user asks about the document or uploaded file.
        """
        retriever = _get_retriever(thread_id)
        if retriever is None:
            return {
                "error": "No document indexed for this chat. Please upload a PDF first.",
                "query": query,
            }
        results = retriever.invoke(query)
        return {
            "query": query,
            "context": [doc.page_content for doc in results],
            "metadata": [doc.metadata for doc in results],
            "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
        }

    return rag_tool


static_tools = [search_tool, get_stock_price, calculator]


# ---------------------------------------------------------------------------
# Forced RAG context injection
#
# Small models often skip tool calls entirely. As a safety net, if a document
# is indexed for the current thread we retrieve relevant chunks BEFORE the LLM
# sees the user message, and inject them directly into the system prompt.
# This means the model gets the document context whether it calls the tool or not.
# ---------------------------------------------------------------------------

def _inject_rag_context(user_query: str, thread_id: str) -> str:
    """
    Retrieve top-k chunks for the query and return them as a formatted string.
    Returns an empty string if no retriever is registered for this thread.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return ""

    try:
        results = retriever.invoke(user_query)
        if not results:
            return ""

        filename = _THREAD_METADATA.get(thread_id, {}).get("filename", "uploaded PDF")
        chunks = "\n\n---\n\n".join(doc.page_content for doc in results)
        return (
            f"\n\n[DOCUMENT CONTEXT from '{filename}']\n"
            f"The following excerpts were retrieved from the uploaded document "
            f"and are relevant to the user's question. Use them to answer accurately.\n\n"
            f"{chunks}\n[END DOCUMENT CONTEXT]"
        )
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def chat_node(state: ChatState, config=None):
    """
    LLM node.
    1. Retrieves RAG context directly and injects it into the system prompt
       (fallback for small models that skip tool calls).
    2. Also binds the rag_tool so capable models can call it explicitly.
    """
    thread_id = ""
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id", "")

    # --- Forced RAG: pull context from the retriever right now ---
    # Find the latest human message to use as the retrieval query
    user_query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and msg.content:
            user_query = str(msg.content)
            break

    rag_context = _inject_rag_context(user_query, thread_id) if user_query else ""

    has_doc = bool(_get_retriever(thread_id))
    doc_instruction = (
        "A document IS indexed for this conversation. "
        "Relevant excerpts have been injected above under [DOCUMENT CONTEXT]. "
        "ALWAYS use that context to answer document-related questions. "
        "You may also call `rag_tool` for additional retrieval if needed."
        if has_doc
        else
        "No document is currently indexed. If the user asks about a PDF, "
        "tell them to upload one using the sidebar."
    )

    system_content = (
        "You are SmartChat Assistant, a helpful AI with access to tools.\n"
        "CRITICAL: Always respond in the same language the user wrote in. "
        "If the user writes in English, respond in English only.\n"  # ADD THIS
        "IMPORTANT TOOL USE RULES:\n"
        "- If the user asks about current events, news, prices, sports, weather, "
        "or ANYTHING that requires up-to-date information → you MUST call search_tool.\n"
        "- If the user asks about math or calculations → call calculator.\n"
        "- If the user asks about a stock → call get_stock_price.\n"
        "- If a document is uploaded and the question relates to it → call rag_tool.\n"
        "Never answer from memory when a tool would give a better answer. "
        "If a tool returns no results or an error, report that honestly — do NOT make up an answer.\n\n"
        f"{doc_instruction}\n"
        f"{rag_context}"
    )


    system_message = SystemMessage(content=system_content)

    # Build tools — bind rag_tool even for small models as a best-effort
    rag_tool_for_thread = make_rag_tool(thread_id)
    current_tools = [*static_tools, rag_tool_for_thread]
    current_llm = llm.bind_tools(current_tools)

    messages = [system_message, *state["messages"]]
    response = current_llm.invoke(messages, config=config)
    return {"messages": [response]}


def dynamic_tool_node(state: ChatState, config=None):
    """Tool executor — uses the thread-specific rag_tool."""
    thread_id = ""
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id", "")

    rag_tool_for_thread = make_rag_tool(thread_id)
    node = ToolNode([*static_tools, rag_tool_for_thread])
    return node.invoke(state, config)


# ---------------------------------------------------------------------------
# Checkpointer
# ---------------------------------------------------------------------------

db_url = os.getenv("NEON_DB_URL")

checkpointer_conn = psycopg.connect(db_url, autocommit=True, prepare_threshold=0)
checkpointer = PostgresSaver(checkpointer_conn)

checkpointer.setup()

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", dynamic_tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def retrieve_all_threads() -> list:
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_docuemnt_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})

def delete_thread_from_db(thread_id: str) -> None:
    """Permanently delete all checkpoint data for a thread from Postgres."""
    tid = str(thread_id)
    with checkpointer_conn.cursor() as cur:
        cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (tid,))
        cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (tid,))
        cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (tid,))
    # autocommit=True so no explicit commit needed