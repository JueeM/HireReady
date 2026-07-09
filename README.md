# HireReady
AI-powered job application assistant — multi-agent RAG pipeline using LangGraph, ChromaDB and Groq that tailors your resume, scores ATS compatibility, and generates cover letters grounded in your actual experience.

HireReady is an end-to-end multi-agent AI system that takes any resume and job description as input and automatically generates a tailored resume, ATS compatibility score, personalized cover letter, and likely interview questions — all grounded in the user's actual experience via a RAG (Retrieval-Augmented Generation) pipeline, with zero hallucination.
The core pipeline is orchestrated using LangGraph and consists of 5 specialized agents. The Gap Analyzer agent compares the uploaded resume against the job description and produces a structured report of matching skills, missing requirements, and repositionable experience. The Tailoring Agent and Resume Critic Agent then run in parallel via LangGraph's parallel branch execution — the Tailoring Agent rewrites resume bullets using JD-aligned keywords drawn exclusively from retrieved resume chunks, while the Resume Critic Agent scores the resume's compatibility with the specific job description. Both agents feed into a merge node, after which the Cover Letter Agent generates a personalized, grounded cover letter, and finally the Interview Question Generator produces 5 role-specific questions the candidate should prepare for.
The RAG pipeline uses PyPDF2 to extract resume text, splits it into semantic chunks via LangChain's RecursiveCharacterTextSplitter, embeds each chunk using sentence-transformers (all-MiniLM-L6-v2), and stores them in ChromaDB. Every agent retrieves only the most relevant chunks before generating output, ensuring responses are grounded solely in verified resume content. Multi-user support is implemented via session-isolated ChromaDB collections — each uploaded resume gets its own unique collection keyed to the user's session ID, so no two users share or overwrite each other's data.
The Streamlit interface allows any user to upload their own resume PDF, paste a job description, and receive the full application package in under 20 seconds. Results are displayed in a structured layout with a downloadable .txt export.


## 🏗️ Architecture

```
User uploads resume PDF
          ↓
Session-isolated ChromaDB (RAG pipeline)
          ↓
    Gap Analyzer Agent
          ↓
  ┌───────────────────┐
  ↓                   ↓
Tailoring Agent    Resume Critic Agent   ← parallel via LangGraph
  ↓                   ↓
  └───────────────────┘
          ↓
   Cover Letter Agent
          ↓
Interview Question Generator
          ↓
   Streamlit UI + Download
```

---

## ✨ Features

- **Multi-user support** — each uploaded resume gets a session-isolated ChromaDB collection
- **RAG pipeline** — all outputs grounded in retrieved resume chunks, no hallucination
- **Parallel execution** — Tailoring and Resume Critic agents run simultaneously (20.8% faster than sequential)
- **ATS scoring** — measures resume-to-JD match before and after tailoring
- **Full application package** — gap report, tailored bullets, cover letter, interview questions, downloadable as .txt

---

## 📊 Results

| Metric | Value |
|---|---|
| ATS score improvement | 70.2 → 83.4 (+13.2 pts avg across 5 JDs) |
| Parallel vs sequential speedup | 20.8% faster |
| End-to-end generation time | < 20 seconds |

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Agent orchestration | LangGraph |
| LLM | Groq API (llama-3.1-8b-instant) |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| PDF parsing | PyPDF2 |
| UI | Streamlit |

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/HireReady.git
cd HireReady
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your API key**

Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Free key at: https://console.groq.com

**5. Run**
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
HireReady/
├── app.py                  ← Streamlit UI
├── pipeline.py             ← LangGraph orchestration
├── requirements.txt
├── .env                    ← API key (never push this)
├── .gitignore
└── rag/
    ├── loader.py           ← PDF text extraction
    ├── chunker.py          ← text splitting
    ├── embedder.py         ← ChromaDB embedding
    ├── gap_analyzer.py     ← Gap Analyzer agent
    └── tailoring_agent.py  ← Tailoring, Critic, Cover Letter, Interview agents
```

---

## ⚠️ Disclaimer

HireReady only rewrites experience that exists in your uploaded resume. It does not invent skills or fabricate experience — every output is grounded in retrieved resume content via RAG.

Tech stack: LangGraph, Groq API (llama-3.1-8b-instant), ChromaDB, sentence-transformers, PyPDF2, Streamlit, python-dotenv.
