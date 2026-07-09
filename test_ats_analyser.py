import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag"))

from loader import load_resume
from chunker import chunk_text
from embedder import embed_and_store
from tailoring_agent import resume_critic_agent, tailoring_agent
from gap_analyzer import analyze_gap

TEST_COLLECTION = "ats_test_collection"
RESUME_PATH = "test_resume.pdf"  # any resume PDF placed in this folder

if not os.path.exists(RESUME_PATH):
    print(f"ERROR: {RESUME_PATH} not found. Place a resume PDF named 'test_resume.pdf' in this folder.")
    sys.exit(1)

print(f"Indexing resume from {RESUME_PATH}...")
text = load_resume(RESUME_PATH)
chunks = chunk_text(text)
embed_and_store(chunks, collection_name=TEST_COLLECTION)

job_descriptions = [
    """Data Science Intern needed with Python, SQL, scikit-learn,
    TensorFlow, data visualization, NLP experience preferred.""",

    """Looking for a Machine Learning Intern with strong skills in
    Python, statistical analysis, model deployment, and experience
    with REST APIs and cloud platforms.""",

    """Business Analyst Intern role requiring Excel, Tableau, SQL,
    stakeholder communication, and data storytelling skills.""",

    """AI Engineer Intern position requiring experience with LLMs,
    RAG pipelines, vector databases, and agent frameworks like
    LangChain or LangGraph.""",

    """Data Analyst Intern needed with strong Python, pandas,
    data cleaning, statistical analysis, and dashboard creation
    using Power BI or Tableau."""
]

before_scores = []
after_scores = []

for i, jd in enumerate(job_descriptions, 1):
    print(f"\n--- Job Description {i} ---")

    before = resume_critic_agent(jd, TEST_COLLECTION)
    before_score = before.get("ats_score", 0)
    before_scores.append(before_score)
    print(f"BEFORE tailoring ATS score: {before_score}")

    gap_report = analyze_gap(jd, TEST_COLLECTION)
    tailored = tailoring_agent(jd, gap_report, TEST_COLLECTION)
    bullets = tailored.get("tailored_bullets", [])

    embed_and_store(bullets, collection_name=f"{TEST_COLLECTION}_after")
    after = resume_critic_agent(jd, f"{TEST_COLLECTION}_after")
    after_score = after.get("ats_score", 0)
    after_scores.append(after_score)
    print(f"AFTER tailoring ATS score: {after_score}")

avg_before = sum(before_scores) / len(before_scores)
avg_after = sum(after_scores) / len(after_scores)

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Average BEFORE score: {avg_before:.1f}")
print(f"Average AFTER score:  {avg_after:.1f}")
print(f"Improvement: +{avg_after - avg_before:.1f} points across {len(job_descriptions)} job descriptions")