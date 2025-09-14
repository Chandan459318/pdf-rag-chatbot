import os
import io
import re
import shutil
import streamlit as st

from dotenv import load_dotenv

# ---------- Robust PDF extraction ----------
import fitz  # PyMuPDF
import pdfplumber
from PyPDF2 import PdfReader

# ---------- LangChain core ----------
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------- Embeddings / LLMs ----------
#from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- Optional: web search (guarded import) ----------
try:
    from duckduckgo_search import DDGS  # pip install duckduckgo-search
    HAVE_DDG = True
except Exception:
    DDGS = None
    HAVE_DDG = False


# =========================
# 🔐 Secrets / Config
# =========================
def load_api_key():
    # Prefer local .env for dev; fallback to Streamlit Secrets in cloud
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None


OPENAI_API_KEY = load_api_key()
if not OPENAI_API_KEY:
    st.warning("No OpenAI API key found. Add it to a local `.env` or Streamlit **Secrets**.")

OPENAI_MODEL = "gpt-4o-mini"            # change to "gpt-4o" or "gpt-3.5-turbo" if you prefer
EMBED_MODEL = "text-embedding-3-small"  # OpenAI embeddings
PERSIST_DIR = "faiss_index"             # FAISS index folder


# =========================
# 🧰 Helpers
# =========================
def extract_documents_from_pdfs(uploaded_files) -> list[Document]:
    """
    Robustly extract text from PDFs.
    For each page, try: PyMuPDF -> pdfplumber -> PyPDF2.
    Returns a list of LangChain Documents with metadata {source, page}.
    """
    docs: list[Document] = []

    for uf in uploaded_files:
        file_bytes = uf.read()
        file_name = getattr(uf, "name", "uploaded.pdf")

        extracted_per_page: list[str] = []

        # 1) PyMuPDF (best at messy PDFs)
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
                for page in pdf:
                    txt = page.get_text("text") or ""
                    extracted_per_page.append(txt)
        except Exception:
            extracted_per_page = []

        # 2) pdfplumber fallback
        if sum(len(t or "") for t in extracted_per_page) < 300:
            try:
                tmp = []
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages:
                        txt = page.extract_text() or ""
                        tmp.append(txt)
                extracted_per_page = tmp
            except Exception:
                pass

        # 3) PyPDF2 last-resort
        if sum(len(t or "") for t in extracted_per_page) < 300:
            try:
                tmp = []
                reader = PdfReader(io.BytesIO(file_bytes))
                for page in reader.pages:
                    txt = page.extract_text() or ""
                    tmp.append(txt)
                extracted_per_page = tmp
            except Exception:
                pass

        # Build Documents with metadata
        for idx, page_text in enumerate(extracted_per_page, start=1):
            page_text = (page_text or "").strip()
            if not page_text:
                continue
            docs.append(Document(page_content=page_text, metadata={"source": file_name, "page": idx}))

    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Chunk while preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
    )
    chunked: list[Document] = []
    for d in docs:
        for c in splitter.split_text(d.page_content):
            chunked.append(Document(page_content=c, metadata=d.metadata))
    return chunked


def build_vectorstore_from_docs(
    docs: list[Document],
    api_key: str,
    force_hf: bool = False
) -> FAISS | None:
    """Create FAISS from Document list. Try OpenAI → fallback/force HF embeddings."""
    if not docs:
        return None

    st.info(f"Indexed {len(docs)} chunks (~{sum(len(d.page_content) for d in docs):,} chars).")
    with st.expander("Preview first chunk"):
        st.write(docs[0].page_content[:1000])

    if force_hf:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.caption("Using Hugging Face embeddings (forced).")
        return FAISS.from_documents(docs, embedding=embeddings)

    # Try OpenAI embeddings first
    try:
        embeddings = OpenAIEmbeddings(api_key=api_key, model=EMBED_MODEL)
        vs = FAISS.from_documents(docs, embedding=embeddings)
        st.success("Built index using OpenAI embeddings.")
        return vs
    except Exception as e:
        st.warning(f"OpenAI embeddings failed ({e.__class__.__name__}). Falling back to Hugging Face.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(docs, embedding=embeddings)
    st.success("Built index using Hugging Face embeddings.")
    return vs


def save_vectorstore(vs: FAISS, path: str = PERSIST_DIR):
    vs.save_local(path)


def load_vectorstore(api_key: str, path: str = PERSIST_DIR) -> FAISS | None:
    if not os.path.isdir(path):
        return None
    # Try to load with OpenAI embeddings, else HF
    try:
        emb = OpenAIEmbeddings(api_key=api_key, model=EMBED_MODEL)
        return FAISS.load_local(path, emb, allow_dangerous_deserialization=True)
    except Exception:
        try:
            emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            return FAISS.load_local(path, emb, allow_dangerous_deserialization=True)
        except Exception:
            return None


def make_retrieval_chain(vs: FAISS, api_key: str):
    # Use a similarity score threshold to detect weak matches
    retriever = vs.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 6, "score_threshold": 0.2},
    )

    llm = ChatOpenAI(
        api_key=api_key,
        model=OPENAI_MODEL,
        temperature=0.1,
        max_tokens=900,
    )

    qa_prompt = PromptTemplate.from_template(
        "You are a helpful assistant answering strictly from the provided context. "
        "If the user’s question could refer to more than one uploaded document, explicitly name which document "
        "you are answering from (use the `source` metadata). "
        "If the answer is not present, reply exactly: \"I don't know based on the documents.\"\n\n"
        "Question: {question}\n\nContext:\n{context}\n\nAnswer:"
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
    )
    return chain


# ----- simple calculator (optional tool) -----
def safe_calculator(expr: str) -> str:
    """Safe eval for simple math: digits, + - * / ( ) . and spaces."""
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expr):
        return "Calculator supports only numbers and + - * / ( )"
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Calc error: {e}"


# ----- web search helpers -----
def web_search_results(query: str, max_results: int = 4):
    """Return a list of dicts: {title, snippet, url}. Empty list if tool missing or no hits."""
    if not HAVE_DDG:
        return []
    try:
        out = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                out.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href") or r.get("url") or "",
                })
        return out
    except Exception:
        return []


def says_idk(text: str) -> bool:
    """Detect any 'I don't know'-like phrasing."""
    t = (text or "").lower()
    patterns = [
        "i don't know", "i do not know",
        "not in the documents", "not present in the documents",
        "no information in the documents",
        "can't find", "cannot find", "couldn't find",
        "based on the documents"
    ]
    return any(p in t for p in patterns)


def ensure_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore_ready" not in st.session_state:
        st.session_state.vectorstore_ready = False
    if "force_hf" not in st.session_state:
        st.session_state.force_hf = False
    if "enable_web_search" not in st.session_state:
        st.session_state.enable_web_search = False


# =========================
# 🖥️ UI
# =========================
st.set_page_config(page_title="📄 PDF Q&A Chatbot", page_icon="🤖", layout="wide")
st.title("📄 PDF Q&A Chatbot — RAG + Chat History")
st.caption("Multi-PDF retrieval with sources. Optional web-search & calculator fallbacks. OpenAI with HF fallback.")

ensure_session_state()

with st.sidebar:
    st.header("Your Documents")
    uploaded_files = st.file_uploader("Upload PDF(s) to index", type="pdf", accept_multiple_files=True)

    st.toggle("Force HuggingFace embeddings (no OpenAI)", key="force_hf")
    if HAVE_DDG:
        st.toggle("Enable Web Search fallback", key="enable_web_search")
    else:
        st.session_state.enable_web_search = False
        st.caption("Web search tool not installed. Run: pip install duckduckgo-search")

    colA, colB = st.columns(2)
    with colA:
        rebuild = st.button("Build / Rebuild Index", type="primary")
    with colB:
        clear_idx = st.button("Clear Index")

    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.success("Chat cleared.")

    st.markdown("---")
    st.subheader("Index Status")
    if os.path.isdir(PERSIST_DIR):
        st.success("FAISS index found ✅")
    else:
        st.info("No index yet. Upload PDFs and click **Build / Rebuild Index**.")

    st.markdown("---")
    st.caption("Tip: On Streamlit Cloud, set **OPENAI_API_KEY** in Secrets.")


# ---------- Index actions ----------
if clear_idx:
    shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    st.session_state.vectorstore_ready = False
    st.success("Index cleared.")

if rebuild:
    if not uploaded_files:
        st.warning("Please upload at least one PDF before rebuilding the index.")
    else:
        with st.spinner("Extracting text & building embeddings..."):
            base_docs = extract_documents_from_pdfs(uploaded_files)
            if not base_docs:
                st.error("No text extracted. This PDF might be scanned. (We can add OCR if needed.)")
            else:
                docs = chunk_documents(base_docs)
                vs = build_vectorstore_from_docs(docs, OPENAI_API_KEY or "", force_hf=st.session_state.force_hf)
                if vs is None:
                    st.error("Failed to build index.")
                else:
                    save_vectorstore(vs, PERSIST_DIR)
                    st.session_state.vectorstore_ready = True
                    st.success("Index built and saved ✅")

# Load index on startup if present
if not st.session_state.vectorstore_ready:
    if load_vectorstore(OPENAI_API_KEY or ""):
        st.session_state.vectorstore_ready = True


# ---------- Chat UI ----------
st.markdown("## Chat")

if st.session_state.vectorstore_ready:
    vs = load_vectorstore(OPENAI_API_KEY or "")
    chain = make_retrieval_chain(vs, OPENAI_API_KEY or "")

    for m in st.session_state.messages:
        st.chat_message(m["role"]).write(m["content"])

    user_q = st.chat_input("Ask your question about the uploaded PDFs...")
    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})
        st.chat_message("user").write(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # 1) Try RAG
                result = chain({"query": user_q})
                answer = (result.get("result") or "").strip()
                sources = result.get("source_documents") or []
                total_ctx_len = sum(len((d.page_content or "")) for d in sources)

                # 2) Decide if fallback is needed
                need_web = False
                need_calc = False
                if total_ctx_len < 300 or says_idk(answer):
                    # If it's a math-like input, prefer calculator; else web (if enabled)
                    if re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", user_q):
                        need_calc = True
                    elif st.session_state.get("enable_web_search", False):
                        need_web = True

                used_web = False
                web_items = []

                # 3) Calculator fallback
                if need_calc:
                    calc_result = safe_calculator(user_q)
                    followup = (
                        "Use this calculator result to answer the user's query succinctly. "
                        "If still unsure, say 'I don't know.'\n\n"
                        f"Calculator result: {calc_result}\nUser question: {user_q}\nAnswer:"
                    )
                    llm = ChatOpenAI(
                        api_key=OPENAI_API_KEY or "",
                        model=OPENAI_MODEL,
                        temperature=0.1,
                        max_tokens=300,
                    )
                    answer = (llm.invoke(followup).content or "").strip()

                # 4) Web-search fallback
                if need_web:
                    st.info("🔎 Web search fallback enabled — looking up additional context...")
                    web_items = web_search_results(user_q, max_results=4)
                    if web_items:
                        used_web = True
                        web_ctx = "\n".join(
                            [f"- {w['title']}: {w['snippet']}" for w in web_items if w["snippet"]]
                        )
                        followup_prompt = (
                            "Use the additional context from the web below to answer the user's question. "
                            "Be concise. If still unsure, say 'I don't know.'\n\n"
                            f"Additional context:\n{web_ctx}\n\n"
                            f"User question: {user_q}\nAnswer:"
                        )
                        llm = ChatOpenAI(
                            api_key=OPENAI_API_KEY or "",
                            model=OPENAI_MODEL,
                            temperature=0.1,
                            max_tokens=700,
                        )
                        answer = (llm.invoke(followup_prompt).content or "").strip()
                    else:
                        st.caption("No web results found for this query.")

                # 5) Render final answer
                st.write(answer if answer else "I don't know based on the documents.")

                # 6) PDF sources
                if sources:
                    with st.expander("Show sources"):
                        for i, doc in enumerate(sources, start=1):
                            meta = doc.metadata or {}
                            src = meta.get("source", "PDF")
                            page = meta.get("page", "?")
                            snippet = (doc.page_content or "").strip().replace("\n", " ")
                            if len(snippet) > 350:
                                snippet = snippet[:350] + "..."
                            st.markdown(f"**Source {i}:** `{src}` — page {page}")
                            st.write(snippet)

                # 7) Web sources
                if used_web and web_items:
                    with st.expander("Web sources (fallback)"):
                        for w in web_items:
                            title = w["title"] or "(no title)"
                            url = w["url"]
                            if url:
                                st.markdown(f"- [{title}]({url})")
                            else:
                                st.markdown(f"- {title}")

        st.session_state.messages.append({"role": "assistant", "content": answer or "I don't know based on the documents."})
else:
    st.info("Upload PDFs and click **Build / Rebuild Index** to start chatting.")
