from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import src.logger as log

lg = log.get("rag_chain")

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a legal assistant helping users understand Malaysian legal documents, "
        "such as the Contracts Act 1950. Answer the question based ONLY on the provided context. "
        "Do not use any outside knowledge. If the answer is not found in the context, reply with "
        "\"Not specified in the document.\" Avoid guessing or making assumptions.\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION: {question}\n"
        "ANSWER:"
    ),
)

def build_qa(retriever, api_key: str):
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0
    )
    lg.info("QA chain initialized ✅")
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
