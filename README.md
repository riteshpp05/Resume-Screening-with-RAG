
# 🧠 Resume Screening with RAG (Streamlit + Gemini)

🔗 Live Demo

https://resume-screening-with-rag-xoysv5tozanstu2rauwkzg.streamlit.app/

This project is a simple **AI-powered Resume Screening Tool** built for the JobTalk backend assessment idea.

It lets a recruiter:

- Upload **one Resume** and **one Job Description** (PDF/TXT)  
- Get a **Match Score (0–100%)**  
- See **Strengths**, **Gaps**, and a **Candidate Summary**  
- Ask questions in a **chat interface** powered by **RAG (Retrieval-Augmented Generation)** over the Resume + JD

Everything is implemented in **Python** using **Streamlit** as the frontend and backend in a single app.

---

## 🚀 Features

- Upload **two documents**:
  - Resume (PDF/TXT)
  - Job Description (PDF/TXT)
- Extracts text using **PyPDF2** (for PDFs) or direct read (for TXT)
- Splits text into chunks using **RecursiveCharacterTextSplitter**
- Builds a local **FAISS** vector store with **HuggingFace embeddings**
- Uses **Google Gemini (via langchain-google-genai)** for:
  - Detailed candidate analysis (summary, skills, experience, strengths, gaps, verdict)
  - RAG-based Q&A chat over Resume + JD
- Simple **Streamlit UI**:
  - Left: Match Score + Full Analysis
  - Right: Chatbox and Chat History

---

## 🧱 Tech Stack

- **Language:** Python 3.10+ (recommended)
- **Frontend + Backend:** Streamlit
- **LLM:** Google Gemini (via `langchain-google-genai`)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- **Vector Store:** FAISS (`langchain_community.vectorstores.FAISS`)
- **PDF Reading:** PyPDF2
- **Env Management:** python-dotenv

---

## 📂 Project Structure

Example structure:

```bash
project-root/
│
├── app.py                         # Main Streamlit app (RAG + UI)
├── job_description_ai_engineer.txt# Sample Job Description file
├── requirements.txt               # Python dependencies
└── .env                           # Environment variables (NOT committed to Git)
```

Example `requirements.txt`:

```txt
streamlit
python-dotenv
langchain-community
langchain-google-genai
sentence-transformers
faiss-cpu
PyPDF2
```

---

## ✅ Prerequisites

- Python 3.10+ installed
- A **Google Gemini API key**

---

## 🧪 Setup & Run (Step-by-Step with Virtual Environment)

### 1️⃣ Clone the repository (or create project folder)

If using Git:

```bash
git clone <your-repo-url>.git
cd <your-repo-folder>
```

---

### 2️⃣ Create and activate a virtual environment

#### 🔹 On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### 🔹 On macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Create `.env` file

```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

---

### 5️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🧑‍💻 How to Use the App

1. Upload **exactly two files**:
   - Resume
   - Job Description
2. Click **Submit & Analyze**
3. View:
   - Match Score
   - Candidate Summary
   - Skills
   - Experience
   - Education
   - Strengths
   - Gaps
   - Final Verdict
4. Ask questions in chat section (RAG powered)

---

## 🧹 Stop & Exit

To stop:

```
CTRL + C
```

To deactivate venv:

```
deactivate
```

---

## 🙋‍♂️ Contact

**Author:** Ritesh Patil  
Machine Learning / AI Engineer  
