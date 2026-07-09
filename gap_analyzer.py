import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import json
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="C:/Users/Admin/Desktop/PROJECTS/HireReady/.env")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")

def retrieve_resume_context(query,collection_name):
    collection=client.get_collection(collection_name)
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=4
    )
    return "\n\n".join(results["documents"][0])

def analyze_gap(job_description,collection_name):
    context = retrieve_resume_context(job_description,collection_name)

    prompt = f"""You are an expert career coach analyzing a candidate's resume against a job description.

Resume content:
{context}

Job Description:
{job_description}

Analyze the resume against the job description and respond ONLY with a JSON object in this exact format, no extra text:
{{
  "have": ["skill or experience the candidate has that matches JD"],
  "missing": ["skill or requirement in JD not found in resume"],
  "reposition": ["skill in resume that could be reframed to match JD better"]
}}
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content

    try:
        cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
        gap_report = json.loads(cleaned)
    except json.JSONDecodeError:
        print("Raw output:\n", raw)
        gap_report = {"error": "Could not parse response as JSON"}

    return gap_report

