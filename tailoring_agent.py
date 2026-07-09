import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv(dotenv_path="C:/Users/Admin/Desktop/PROJECTS/HireReady/.env")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")

def retrieve_resume_context(query,collection_name):
    collection=client.get_collection(collection_name)
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5
    )
    return "\n\n".join(results["documents"][0])

#agent 2- tailoring agent
def tailoring_agent(job_description,gap_report,collection_name):
    context=retrieve_resume_context(job_description,collection_name)
    prompt = f"""You are an expert resume writer.
Using ONLY the resume content below, rewrite 4-5 strong bullet points
that are tailored to the job description.

Rules:
- Only use experience that exists in the resume
- Use keywords from the job description naturally
- Start each bullet with a strong action verb
- Quantify impact wherever possible
- Do not invent anything
Resume content:
{context}

Job Description:
{job_description}

Skills the candidate has that match JD:
{gap_report.get("have", [])}

Skills to reposition:
{gap_report.get("reposition", [])}
Respond ONLY with a JSON object in this exact format, no extra text:
{{
  "tailored_bullets": [
    "bullet 1",
    "bullet 2",
    "bullet 3",
    "bullet 4",
    "bullet 5"
  ]
}}
"""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content
    try:
        cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print("Raw output:\n", raw)
        return {"tailored_bullets:[]"}
    
#agent:resume critic(ATS score)-runs in parellel with tailoring
def resume_critic_agent(job_description, collection_name):
    context = retrieve_resume_context(job_description, collection_name)

    prompt = f"""You are an ATS (Applicant Tracking System) expert.
Score how well this resume content matches the SPECIFIC job description below — not generic resume quality.

Resume content:
{context}

Job Description:
{job_description}

Score based on:
- How many of the JD's required skills/keywords appear in this resume content
- How directly the resume's experience maps to this specific role
- Whether action verbs and quantified results align with what this JD values

Respond ONLY with a JSON object:
{{
  "ats_score": <integer 0 to 100, reflecting match to THIS job description specifically>,
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "quick_fixes": ["fix 1", "fix 2"]
}}
"""
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content
    try:
        cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print("Raw output:\n", raw)
        return {"ats_score": 0, "strengths": [], "weaknesses": [], "quick_fixes": []}



#agent-cover letter
def cover_letter_agent(job_description, tailored_bullets, company_name):
    bullets_text = "\n".join(tailored_bullets)

    prompt = f"""You are an expert cover letter writer.
Write a professional 3-paragraph cover letter for the following:

Company: {company_name}
Job Description: {job_description}

Candidate's tailored experience:
{bullets_text}

Rules:
- Paragraph 1: why this role and this company excite the candidate
- Paragraph 2: 2-3 most relevant experiences from the bullets above
- Paragraph 3: forward looking close, express enthusiasm
- Keep it under 250 words
- Professional but not robotic tone
- Do not invent any experience

Respond with just the cover letter text, no labels or headings.
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

#agent-interview question generator
def interview_question_agent(job_description, tailored_bullets):
    bullets_text = "\n".join(tailored_bullets)

    prompt = f"""You are an interview coach.
Based on the job description and the candidate's tailored resume bullets below,
generate 5 likely interview questions the candidate should prepare for.

Job Description:
{job_description}

Candidate's experience:
{bullets_text}
Mix of question types:
- 2 technical/skill-based questions
- 2 behavioral questions tied to their actual projects
- 1 "why this role/company" question

Respond ONLY with a JSON object in this exact format, no extra text:
{{
  "questions": ["question 1", "question 2", "question 3", "question 4", "question 5"]
}}
"""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content
    try:
        cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print("Raw output:\n", raw)
        return {"questions": []}

