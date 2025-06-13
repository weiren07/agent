import os, io, pathlib, hashlib, tempfile, traceback
import streamlit as st
from dotenv import load_dotenv

# ── ENV ───────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("❌ GOOGLE_API_KEY missing in .env"); st.stop()

# ── Core imports ───────────────────────────────────────────────
from src import retriever, rag_chain, checklist_lib
from src.intent_router import classify_intent

# ── UI setup ───────────────────────────────────────────────────
st.set_page_config(page_title="⚖️ Agentic RAG – demo", page_icon="⚖️", layout="centered")
st.title("⚖️ Agentic RAG – demo")
st.caption("Upload a PDF once, then ask questions or type **checklist: ...**")

# ── Session-state ─────────────────────────────────────────────
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ── Cached builder (temp file used) ───────────────────────────
@st.cache_resource(hash_funcs={bytes: lambda b: hashlib.md5(b).hexdigest()})
def build_qa_chain(pdf_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    vect = retriever.build_index(tmp_path, API_KEY)
    return rag_chain.build_qa(vect, API_KEY)

# ── File upload ───────────────────────────────────────────────
pdf_file = st.file_uploader("📄 Upload PDF", type="pdf", key="pdf")
if pdf_file and st.session_state.qa_chain is None:
    pdf_bytes = pdf_file.getvalue()
    with st.spinner("🔄 Indexing …"):
        st.session_state.qa_chain = build_qa_chain(pdf_bytes)
    st.success("✅ Index ready!")

if st.session_state.qa_chain is None:
    st.stop()

# ── Prompt + Logic ─────────────────────────────────────────────
def answer_question(q: str) -> str:
    res = st.session_state.qa_chain.invoke({"query": q})
    pages = sorted({d.metadata.get("page") for d in res["source_documents"] if d.metadata.get("page")})
    txt = res["result"]
    if pages:
        txt += f"\n\n📑 Pages: {', '.join(map(str, pages))}"
    return txt

def fill_checklist():
    default_template = pathlib.Path("document/law.xlsx")
    return checklist_lib.fill_template(str(default_template), st.session_state.qa_chain)

with st.form("ask_form"):
    prompt = st.text_input("📝 Prompt", placeholder="e.g. 'checklist: confidentiality clause'")
    submitted = st.form_submit_button("🚀 Go")

if submitted and prompt.strip():
    try:
        intent = classify_intent(prompt)

        if intent == "CHECKLIST":
            with st.spinner("📑 Building checklist …"):
                output_path = fill_checklist()
            st.success("✅ Checklist ready!")
            with open(output_path, "rb") as f:
                st.download_button("⬇️ Download checklist",
                                   data=f,
                                   file_name=pathlib.Path(output_path).name,
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            with st.spinner("💬 Answering …"):
                answer = answer_question(prompt)
            st.markdown("#### 📄 Answer")
            st.write(answer)

    except Exception:
        st.error("⚠️ Something went wrong — check logs.")
        traceback.print_exc()
