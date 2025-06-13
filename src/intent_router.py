"""
intent_router.py
────────────────
Only classifies the user's intent.
(No indexing – that is the caller's job.)
"""
import os, sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY") or sys.exit("❌ GOOGLE_API_KEY missing")

# ── LLM router -----------------------------------------------------------
clf_llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    google_api_key=API_KEY,
    temperature=0,
)
CLF_PROMPT = """
You are an intent classifier.

Your task is to analyze the user message and return ONLY ONE WORD based on the intent:
- Return **CHECKLIST** if the message indicates the user wants to generate a summary or a checklist (e.g., "generate summary", "checklist for termination clause", "summarize this contract").
- Return **QUESTION** for any other inquiry, such as definitions, explanations, or factual questions from the document.

Do not explain your choice. Only output one of the following:
CHECKLIST
QUESTION

Message: "{msg}"

Answer:
"""

def classify_intent(msg: str) -> str:
    label = clf_llm.invoke(CLF_PROMPT.format(msg=msg)).content.strip().upper()
    return "CHECKLIST" if "CHECKLIST" in label else "QUESTION"

# ------------------------------------------------------------------------
# Optional CLI test tool (runs ONLY when this file is executed directly)
# ------------------------------------------------------------------------
if __name__ == "__main__":
    import pathlib, hashlib, time
    from src import retriever, rag_chain, checklist_lib

    PDF_PATH = "document/Contracts-Act-1950.pdf"
    XLS_PATH = "document/law.xlsx"

    print("🔄 Indexing sample PDF once for CLI demo …")
    vect = retriever.build_index(PDF_PATH, API_KEY)
    qa   = rag_chain.build_qa(vect, API_KEY)

    def answer(q):
        res = qa({"query": q})
        pages = sorted({d.metadata.get("page") for d in res["source_documents"] if d.metadata.get("page")})
        return res["result"] + (f"\n📑 Pages: {pages}" if pages else "")

    print("Agentic RAG CLI — type 'exit' to quit")
    while True:
        text = input(">> ").strip()
        if text.lower() in {"exit", "quit"}:
            break
        if classify_intent(text) == "CHECKLIST":
            path = checklist_lib.fill_template(XLS_PATH, qa)
            print("Filled checklist:", path)
        else:
            print(answer(text))
