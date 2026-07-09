import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag"))

from loader import load_resume
from chunker import chunk_text
from embedder import embed_and_store
from tailoring_agent import tailoring_agent, resume_critic_agent
from gap_analyzer import analyze_gap

TEST_COLLECTION = "ats_test_collection"
RESUME_PATH = "test_resume.pdf"

# index a fresh copy if not already indexed
if not os.path.exists(RESUME_PATH):
    print(f"ERROR: {RESUME_PATH} not found. Place a resume PDF named 'test_resume.pdf' in this folder.")
    sys.exit(1)

print(f"Indexing resume from {RESUME_PATH}...")
text = load_resume(RESUME_PATH)
chunks = chunk_text(text)
embed_and_store(chunks, collection_name=TEST_COLLECTION)

job_description = """Data Science Intern needed with Python, SQL,
scikit-learn, TensorFlow, data visualization, NLP experience preferred."""

gap_report = analyze_gap(job_description, TEST_COLLECTION)

print("\nRunning SEQUENTIAL (one after another)...")
start = time.time()

tailored = tailoring_agent(job_description, gap_report, TEST_COLLECTION)
ats = resume_critic_agent(job_description, TEST_COLLECTION)

sequential_time = time.time() - start
print(f"Sequential time: {sequential_time:.2f} seconds")

import concurrent.futures

print("\nRunning PARALLEL (simultaneous)...")
start = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    future_tailor = executor.submit(tailoring_agent, job_description, gap_report, TEST_COLLECTION)
    future_ats = executor.submit(resume_critic_agent, job_description, TEST_COLLECTION)

    tailored_parallel = future_tailor.result()
    ats_parallel = future_ats.result()

parallel_time = time.time() - start
print(f"Parallel time: {parallel_time:.2f} seconds")

speedup_pct = ((sequential_time - parallel_time) / sequential_time) * 100

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Sequential: {sequential_time:.2f}s")
print(f"Parallel:   {parallel_time:.2f}s")
print(f"Speedup:    {speedup_pct:.1f}% faster with parallel execution")