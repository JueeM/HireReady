from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
import sys
import os
sys.path.insert(0,os.path.dirname(__file__))

from gap_analyzer import analyze_gap
from tailoring_agent import (
    tailoring_agent,
    resume_critic_agent,
    cover_letter_agent,
    interview_question_agent
)

class ApplicationState(TypedDict):
    job_description:str
    company_name:str
    collection_name:str
    gap_report:dict
    tailored_bullets:List[str]
    ats_report:dict
    cover_letter:str
    interview_questions:List[str]

def run_gap_analyzer(state:ApplicationState)->ApplicationState:
    print("gap aalyzer running")
    gap_report=analyze_gap(state['job_description'],state["collection_name"])
    return {"gap_report":gap_report}

def run_tailoring_agent(state:ApplicationState)->ApplicationState:
    print("tailoring agent running")
    result=tailoring_agent(state["job_description"],state["gap_report"],state["collection_name"])
    return{"tailored_bullets":result.get("tailored_bullets",[])} # the [] act as a fallback

def run_resume_critic_agent(state:ApplicationState)->ApplicationState:
    print("resume critic agent(ATS score)")
    ats_report=resume_critic_agent(state["job_description"],state["collection_name"])
    return {"ats_report":ats_report}

def run_cover_letter_agent(state:ApplicationState)->ApplicationState:
    print("cover letter agent running")
    cover_letter=cover_letter_agent(state["job_description"],state["tailored_bullets"],state["company_name"])
    print("COVER LETTER OUTPUT:", cover_letter)
    return {"cover_letter":cover_letter}

def run_interview_question_agent(state:ApplicationState)->ApplicationState:
    print("interview question generator")
    result2=interview_question_agent(state["job_description"],state["tailored_bullets"])
    return {"interview_questions":result2.get("questions",[])}

def merge_node(state:ApplicationState)->ApplicationState:
    return state

def build_pipeline():
    graph=StateGraph(ApplicationState)

    graph.add_node("gap_analyzer",run_gap_analyzer)
    graph.add_node("tailoring_agent",run_tailoring_agent)
    graph.add_node("resume_critic_agent",run_resume_critic_agent)
    graph.add_node("merge",merge_node)
    graph.add_node("cover_letter_agent",run_cover_letter_agent)
    graph.add_node("interview_question_agent",run_interview_question_agent)

    graph.set_entry_point("gap_analyzer")

    #parellel fan out
    graph.add_edge("gap_analyzer","tailoring_agent")
    graph.add_edge("gap_analyzer","resume_critic_agent")

    #fan in
    graph.add_edge("tailoring_agent","merge")
    graph.add_edge("resume_critic_agent","merge")

    #sequence continues
    graph.add_edge("merge","cover_letter_agent")
    graph.add_edge("cover_letter_agent","interview_question_agent")
    graph.add_edge("interview_question_agent",END)

    return graph.compile()

def run_pipeline(job_description: str, company_name: str, collection_name: str) -> dict:
    pipeline = build_pipeline()

    initial_state = ApplicationState(
        job_description=job_description,
        company_name=company_name,
        collection_name=collection_name,
        gap_report={},
        tailored_bullets=[],
        ats_report={},
        cover_letter="",
        interview_questions=[]
    )

    final_state = pipeline.invoke(initial_state)
    return {
        "gap_report": final_state["gap_report"],
        "tailored_bullets": final_state["tailored_bullets"],
        "ats_report": final_state["ats_report"],
        "cover_letter": final_state["cover_letter"],
        "interview_questions": final_state["interview_questions"]
    }
