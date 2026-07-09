# Week 1 prototype — standalone RAG retrieval test on a single resume PDF
# Superseded by gap_analyzer.py and tailoring_agent.py in the full pipeline

import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
groq_client=Groq(api_key=os.getenv("GROQ_API_KEY"))

model=SentenceTransformer("all-MiniLM-L6-v2")
client=chromadb.PersistentClient(path="./chroma_db")
collection=client.get_collection("resume")

def ask_resume(question):
    #1st step=embed the question
    question_embedding=model.encode([question]).tolist()

    results=collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )
    retrieved_chunks=results["documents"][0]
    context="\n\n".join(retrieved_chunks)

    prompt=f"""you are a helpful assistant analyzing a resume.
Use ONLY the resume content below to answer the question.
Do not add any information not present in the resume.

Resume content:
{context}

Question: {question}

Answer:"""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

questions = [
    "What programming languages does this person know?",
    "What projects has this person built?",
    "What is this person's educational background?",
    "What internships or work experience do they have?",
    "What machine learning skills do they have?"
]
for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {ask_resume(q)}")
    print("-" * 50)