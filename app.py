"""
Streamlit frontend for the PDF (Thai recipes) RAG assistant.

Stack: phidata Assistant + Ollama (llama3.2 / nomic-embed-text) + PgVector.

"""
from typing import List

import httpx
import streamlit as st
from dotenv import load_dotenv
from phi.assistant import Assistant
from phi.embedder.ollama import OllamaEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.llm.ollama import Ollama
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.vectordb.pgvector import PgVector2

load_dotenv()

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DEFAULT_DB_URL = "postgresql+psycopg://ai:ai@localhost:5532/ai"
DEFAULT_PDF = "https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"
OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIMS = 768  # must match the pgvector table (OpenAI would be 1536)

EXAMPLES = [
    "How do I make Pad Thai?",
    "Which recipes are vegetarian?",
    "What ingredients go into green curry?",
    "Suggest a quick Thai dessert.",
]

st.set_page_config(page_title="Thai Recipe Assistant", page_icon="🍜", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def list_ollama_models() -> List[str]:
    """Return locally installed Ollama models, or [] if the server is down."""
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2.0)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except Exception:
        return []


@st.cache_resource(show_spinner="Loading knowledge base (first run can take a few minutes)...")
def get_knowledge_base(pdf_url: str, db_url: str, recreate: bool = False) -> PDFUrlKnowledgeBase:
    kb = PDFUrlKnowledgeBase(
        urls=[pdf_url],
        vector_db=PgVector2(
            collection="recipes",
            db_url=db_url,
            embedder=OllamaEmbedder(model=EMBED_MODEL, dimensions=EMBED_DIMS),
        ),
    )
    kb.load(recreate=recreate)
    return kb


@st.cache_resource
def get_storage(db_url: str) -> PgAssistantStorage:
    return PgAssistantStorage(table_name="pdf_assistant", db_url=db_url)


def build_assistant(kb, storage, model: str, user: str) -> Assistant:
    return Assistant(
        user_id=user,
        knowledge_base=kb,
        storage=storage,
        add_references_to_prompt=True,  # RAG context injected into the prompt
        add_chat_history_to_messages=True,  # keep conversational memory
        num_history_messages=6,
        llm=Ollama(model=model),
    )


def reset_chat():
    st.session_state.messages = []
    st.session_state.pop("assistant", None)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")

    models = list_ollama_models()
    if not models:
        st.error("Can't reach Ollama at localhost:11434. Start the Ollama app.", icon="⚠️")
    elif not any(m.startswith(CHAT_MODEL) for m in models):
        st.warning(f"Ollama is running, but `{CHAT_MODEL}` isn't installed. Run `ollama pull {CHAT_MODEL}`.", icon="⚠️")
    else:
        st.success("Ollama is running", icon="✅")

    user_id = st.text_input("User ID", value="user")
    num_sources = st.slider("Source passages to show", 1, 5, 3)
    show_sources = st.toggle("Show sources under answers", value=True)

    with st.expander("Database & document"):
        db_url = st.text_input("Postgres URL", value=DEFAULT_DB_URL)
        pdf_url = st.text_input("PDF URL", value=DEFAULT_PDF)
        rebuild = st.button("Rebuild knowledge base", help="Drops and re-embeds the PDF")

    st.divider()
    if st.button("New chat", use_container_width=True):
        reset_chat()
        st.rerun()

# --------------------------------------------------------------------------- #
# Knowledge base + assistant
# --------------------------------------------------------------------------- #
if rebuild:
    get_knowledge_base.clear()
    try:
        get_knowledge_base(pdf_url, db_url, recreate=True)
        st.sidebar.success("Knowledge base rebuilt.")
    except Exception as e:
        st.sidebar.error(f"Rebuild failed: {e}")

try:
    knowledge_base = get_knowledge_base(pdf_url, db_url)
    storage = get_storage(db_url)
except Exception as e:
    st.error(
        "Could not load the knowledge base. Check that Postgres (Docker) and Ollama are running.\n\n"
        f"`{type(e).__name__}: {e}`"
    )
    st.stop()

# Rebuild the assistant if the model or user changed
config_key = (user_id, pdf_url, db_url)
if st.session_state.get("config_key") != config_key or "assistant" not in st.session_state:
    st.session_state.assistant = build_assistant(knowledge_base, storage, CHAT_MODEL, user_id)
    st.session_state.config_key = config_key

assistant: Assistant = st.session_state.assistant

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.title("🍜 Thai Recipe Assistant")
st.caption(f"Answers come from the Thai cookbook PDF. Model: `{CHAT_MODEL}` · Run: `{assistant.run_id}`")


def render_sources(sources: List[dict]):
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            page = s.get("page")
            label = f"Passage {i}" + (f" (page {page})" if page else "")
            st.markdown(f"**{label}**")
            st.caption(s["content"])


def get_sources(query: str, k: int) -> List[dict]:
    try:
        docs = knowledge_base.search(query=query, num_documents=k)
    except Exception:
        return []
    return [
        {
            "content": (d.content[:500] + "...") if len(d.content) > 500 else d.content,
            "page": (d.meta_data or {}).get("page"),
        }
        for d in docs
    ]


# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            render_sources(msg["sources"])

# Example prompts on empty chat
clicked_example = None
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, use_container_width=True):
            clicked_example = ex

prompt = st.chat_input("Ask about a Thai recipe...") or clicked_example

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(assistant.run(prompt, stream=True))
        except Exception as e:
            answer = ""
            st.error(f"The model call failed: `{type(e).__name__}: {e}`")

        sources: List[dict] = []
        if answer:
            sources = get_sources(prompt, num_sources)
            if sources and show_sources:
                render_sources(sources)
        elif not answer:
            st.warning(
                "The model returned an empty answer. Start a new chat and try again, "
                f"or change `CHAT_MODEL` at the top of app.py (currently `{CHAT_MODEL}`)."
            )

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})