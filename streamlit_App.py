import os
import math

import streamlit as st
from PyPDF2 import PdfReader

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# ----------------- LOAD .ENV -----------------
load_dotenv()  # loads variables from .env into environment


# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Resume Screening with RAG",
    layout="wide"
)

st.title("🧠 Resume Screening with RAG (Python + Streamlit)")


# ----------------- HELPERS -----------------
def get_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error(
            "GOOGLE_API_KEY is not set. Please create a `.env` file with:\n\n"
            "GOOGLE_API_KEY=your_actual_key_here"
        )
        st.stop()
    return api_key


def read_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def read_txt(file) -> str:
    raw = file.read()
    try:
        return raw.decode("utf-8", errors="ignore")
    except AttributeError:
        return str(raw)


def read_file(file) -> str:
    if file is None:
        return ""
    name = file.name.lower()
    if name.endswith(".pdf"):
        return read_pdf(file)
    elif name.endswith(".txt"):
        return read_txt(file)
    else:
        return ""


def split_into_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_text(text)


def build_vectorstore_from_text(resume_text: str, jd_text: str):
    combined_text = (
        "RESUME:\n" + resume_text.strip() + "\n\n" +
        "JOB DESCRIPTION:\n" + jd_text.strip()
    )

    chunks = split_into_chunks(combined_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore, embeddings


def get_llm():
    api_key = get_api_key()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key,
    )
    return llm


def cosine_similarity(a, b) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_match_score(embeddings, resume_text: str, jd_text: str) -> int:
    if not resume_text.strip() or not jd_text.strip():
        return 0
    emb_resume = embeddings.embed_query(resume_text[:5000])
    emb_jd = embeddings.embed_query(jd_text[:5000])
    sim = cosine_similarity(emb_resume, emb_jd)
    score = (sim + 1) / 2 * 100  # map -1..1 -> 0..100
    return int(round(score))


def analyze_resume_with_llm(llm, resume_text: str, jd_text: str, match_score: int) -> str:
    parser = StrOutputParser()

    prompt = f"""
You are a resume screening assistant.

You are given:
- A candidate RESUME
- A JOB DESCRIPTION
- A pre-computed MATCH SCORE between 0 and 100.

Your task:
1. Provide a short **Candidate Summary**.
2. List **Key Skills** (bullet points).
3. Summarize **Experience** (bullet points).
4. Summarize **Education**.
5. List **Strengths vs Job Description** (bullet points).
6. List **Gaps vs Job Description** (missing skills, years of experience, domain, etc.).
7. Give a short **Final Verdict** (e.g., "Strong match for Backend Engineer", etc.).

Return the answer in **clean Markdown** with headings:
- ### Candidate Summary
- ### Key Skills
- ### Experience Overview
- ### Education
- ### Strengths vs Job Description
- ### Gaps vs Job Description
- ### Final Verdict

MATCH SCORE: {match_score} / 100

RESUME:
\"\"\"{resume_text}\"\"\"

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"
"""
    result = (llm | parser).invoke(prompt)
    return result


def get_rag_answer(llm, vectorstore, question: str, chat_history):
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(d.page_content for d in docs)

    history_str = ""
    if chat_history:
        history_str = "\n".join(
            [f"Q: {h['question']}\nA: {h['answer']}" for h in chat_history[-5:]]
        )

    parser = StrOutputParser()

    prompt = f"""
You are a helpful assistant for resume screening.

Use ONLY the information from the context and chat history below.
If the answer is not present, say: "answer is not available in the context".

Context:
\"\"\"{context}\"\"\"

Chat history:
\"\"\"{history_str}\"\"\"

Question:
\"\"\"{question}\"\"\"

Answer:
"""
    result = (llm | parser).invoke(prompt)
    return result


# ----------------- SESSION STATE -----------------
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = None

if "match_score" not in st.session_state:
    st.session_state["match_score"] = None

if "analysis_md" not in st.session_state:
    st.session_state["analysis_md"] = ""

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# ----------------- SIDEBAR: SINGLE UPLOADER -----------------
with st.sidebar:
    st.header("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload **two files**: first Resume, then Job Description (PDF/TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="resume_jd_uploader",
    )

    if uploaded_files:
        st.info(
            "Files uploaded:\n\n" +
            "\n".join([f"- {f.name}" for f in uploaded_files])
        )

    if st.button("Submit & Analyze"):
        api_key = get_api_key()  # just to validate early

        if not uploaded_files or len(uploaded_files) != 2:
            st.error("Please upload exactly **2 files**: first the Resume, then the Job Description.")
        else:
            resume_file = uploaded_files[0]
            jd_file = uploaded_files[1]

            try:
                with st.spinner("Reading and processing documents..."):
                    resume_text = read_file(resume_file)
                    jd_text = read_file(jd_file)

                    if not resume_text.strip():
                        st.error("Could not extract any text from the Resume.")
                        st.stop()
                    if not jd_text.strip():
                        st.error("Could not extract any text from the Job Description.")
                        st.stop()

                    vectorstore, embeddings = build_vectorstore_from_text(
                        resume_text, jd_text
                    )
                    st.session_state["vectorstore"] = vectorstore
                    st.session_state["embeddings"] = embeddings

                    llm = get_llm()

                    match_score = compute_match_score(
                        embeddings, resume_text, jd_text
                    )
                    st.session_state["match_score"] = match_score

                    analysis_md = analyze_resume_with_llm(
                        llm, resume_text, jd_text, match_score
                    )
                    st.session_state["analysis_md"] = analysis_md

                    st.session_state["chat_history"] = []

                st.success("Analysis complete!")
            except Exception as e:
                st.error(f"Error while processing documents: {str(e)}")


# ----------------- MAIN LAYOUT -----------------
col_left, col_right = st.columns([1.1, 1])

# ---- LEFT: MATCH RESULT & ANALYSIS ----
with col_left:
    st.subheader("📊 Match Result")

    if st.session_state["match_score"] is not None:
        score = st.session_state["match_score"]
        st.metric("Match Score", f"{score} / 100")

    if st.session_state["analysis_md"]:
        st.markdown("---")
        st.markdown(st.session_state["analysis_md"])

# ---- RIGHT: CHAT (RAG) ----
with col_right:
    st.subheader("💬 Ask Questions about this Candidate")

    question = st.text_input(
        "Ask a question based on the Resume + Job Description:",
        key="chat_question",
    )

    if question:
        if st.session_state["vectorstore"] is None:
            st.error("Please upload and analyze documents first.")
        else:
            try:
                llm = get_llm()
                with st.spinner("Thinking..."):
                    answer = get_rag_answer(
                        llm,
                        st.session_state["vectorstore"],
                        question,
                        st.session_state["chat_history"],
                    )

                st.session_state["chat_history"].append(
                    {"question": question, "answer": answer}
                )

            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

    if st.session_state["chat_history"]:
        st.markdown("### Chat History")
        for turn in st.session_state["chat_history"]:
            st.markdown(f"**Q:** {turn['question']}")
            st.markdown(f"**A:** {turn['answer']}")
            st.markdown("---")
