from __future__ import annotations

import os
import re
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
# from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool as lc_tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from tavily import TavilyClient
import requests
import psycopg

load_dotenv()

# ---------------------------------------------------------------------------
# LLM + Embeddings
# ---------------------------------------------------------------------------
# LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
LLM_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


# llm = ChatOllama(model=LLM_MODEL, base_url="http://localhost:11434")
llm = ChatGroq(
    model=LLM_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
)
# embeddings = OllamaEmbeddings(model="nomic-embed-text")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

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
    """
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=True,
        )

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
def calculator(first_num: float, second_num: float, operation: str) -> str:
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
                return "Error: division by zero is not allowed."
            result = first_num / second_num
        else:
            return f"Error: unsupported operation '{operation}'. Use add, sub, mul, or div."
        return f"{first_num} {operation} {second_num} = {result}"
    except Exception as e:
        return f"Calculator error: {e}"


@tool
def get_stock_price(symbol: str) -> str:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage.
    """
    api_key = os.getenv("API_KEY")
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    )
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        quote = data.get("Global Quote", {})
        if not quote:
            return f"No data found for symbol '{symbol}'. Check the ticker or API key."
        return (
            f"Stock: {quote.get('01. symbol')}\n"
            f"Price: ${quote.get('05. price')}\n"
            f"Change: {quote.get('09. change')} ({quote.get('10. change percent')})\n"
            f"Volume: {quote.get('06. volume')}\n"
            f"Last trading day: {quote.get('07. latest trading day')}"
        )
    except Exception as e:
        return f"Error fetching stock price: {e}"


# ---------------------------------------------------------------------------
# Dynamic RAG tool factory
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
# ---------------------------------------------------------------------------

def _inject_rag_context(user_query: str, thread_id: str) -> str:
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
# Intent classifier  (Layer 1)
#
# Detects greetings/small talk BEFORE calling the LLM so we can unbind tools.
# Uses fuzzy matching to handle common typos like "hye", "helo", "thansk".
# ---------------------------------------------------------------------------

# Canonical conversational tokens — we'll fuzzy-match against these
_CONVERSATIONAL_TOKENS = [
    "hey", "hi", "hello", "howdy", "hiya", "greetings", "sup", "yo",
    "thanks", "thank you", "ty", "thx", "cheers", "np", "no problem",
    "okay", "ok", "sure", "cool", "great", "nice", "awesome", "got it", "alright",
    "bye", "goodbye", "see you", "later", "cya",
]

# Exact multi-word conversational phrases
_CONVERSATIONAL_PHRASES = re.compile(
    r"^\s*("
    r"what'?s up|how are you|how r u|how are u|how'?s it going|how do you do|"
    r"good morning|good afternoon|good evening|good night|"
    r"what('?s| is) your name|who are you|what are you|"
    r"are you (an? )?ai|are you (an? )?bot|are you (an? )?robot|"
    r"whats up|watsup|wassup"
    r")\s*[!?.]*\s*$",
    re.IGNORECASE,
)


def _fuzzy_match_token(word: str, target: str, max_distance: int = 1) -> bool:
    """Simple Levenshtein-distance check for short words (handles typos)."""
    if abs(len(word) - len(target)) > max_distance:
        return False
    if word == target:
        return True
    # Count character edits (insertions / deletions / substitutions)
    prev = list(range(len(target) + 1))
    for i, cw in enumerate(word):
        curr = [i + 1]
        for j, ct in enumerate(target):
            curr.append(min(prev[j] + (0 if cw == ct else 1),
                            curr[j] + 1,
                            prev[j + 1] + 1))
        prev = curr
    return prev[-1] <= max_distance


def _is_conversational(text: str) -> bool:
    """
    Return True if the message is pure small talk — no tools needed.
    Handles typos (hye → hey, thansk → thanks) via fuzzy matching.
    """
    text = text.strip().rstrip("!?.")

    # Check multi-word phrases first
    if _CONVERSATIONAL_PHRASES.match(text + " "):  # add space so $ anchors correctly
        return True

    # Split into words and check if ALL words fuzzy-match conversational tokens
    words = text.lower().split()
    if not words or len(words) > 4:  # long messages are never just small talk
        return False

    for word in words:
        matched = any(_fuzzy_match_token(word, token.split()[0]) for token in _CONVERSATIONAL_TOKENS)
        if not matched:
            return False
    return True


def _strip_tool_calls(response: AIMessage) -> AIMessage:
    """
    Layer 2 safety net: if the model somehow still emitted tool_calls
    on a conversational message, strip them out so the graph doesn't
    route to the tool node.
    """
    if not getattr(response, "tool_calls", None):
        return response
    # Return a clean AIMessage with only the text content
    clean_content = response.content if isinstance(response.content, str) else ""
    return AIMessage(content=clean_content or "Hey there! How can I help you?")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def chat_node(state: ChatState, config=None):
    thread_id = ""
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id", "")

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
        "CRITICAL: Always respond in the same language the user wrote in.\n\n"
        "TOOL USE RULES:\n"
        "- Only call search_tool when the user explicitly asks about a specific "
        "topic, event, person, news, price, or fact that requires current data.\n"
        "- Only call calculator when the user asks you to compute a math expression.\n"
        "- Only call get_stock_price when the user asks about a specific stock ticker.\n"
        "- Only call rag_tool when the user asks about the uploaded document.\n\n"
        "RESPONSE RULES:\n"
        "- Report tool results clearly and directly.\n"
        "- NEVER fabricate error messages. If a tool succeeded, present its output.\n\n"
        f"{doc_instruction}\n"
        f"{rag_context}"
    )

    system_message = SystemMessage(content=system_content)

    conversational = _is_conversational(user_query)

    if conversational:
        # Layer 1: don't bind tools at all — model physically cannot call them
        current_llm = llm
    else:
        rag_tool_for_thread = make_rag_tool(thread_id)
        current_tools = [*static_tools, rag_tool_for_thread]
        current_llm = llm.bind_tools(current_tools)

    messages = [system_message, *state["messages"]]
    response = current_llm.invoke(messages, config=config)

    if conversational:
        # Layer 2: strip any tool_calls the model hallucinated anyway
        response = _strip_tool_calls(response)

    return {"messages": [response]}


def dynamic_tool_node(state: ChatState, config=None):
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

def retrieve_all_threads(user_id: str) -> list:
    """Return only threads that belong to the given user."""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        meta = checkpoint.metadata or {}
        if meta.get("user_id") == user_id:
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