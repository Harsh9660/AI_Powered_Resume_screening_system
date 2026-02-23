from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import os
from src.utils.parser import parse_pdf
from src.utils.processor import ResumeProcessor
from src.components.scoring_engine import ScoringEngine

app = FastAPI(title="AI-Powered Resume Screening System")

# Loaded once at startup (expensive: spaCy model)
processor = ResumeProcessor()
engine = ScoringEngine()


@app.get("/")
def read_root():
    return {"message": "Welcome to the AI-Powered Resume Screening System"}


@app.post("/rank-resumes")
async def rank_resumes(job_description: str = Form(...), resumes: List[UploadFile] = File(...)):
    results = []

    jd_skills = set(processor.extract_skills(job_description))
    jd_text = job_description  # Use raw text for TF-IDF (richer signal)

    for resume_file in resumes:
        resume_text = parse_pdf(resume_file)

        if not resume_text or not resume_text.strip():
            results.append({
                "filename": resume_file.filename,
                "score": 0.0,
                "skills": [],
                "explanation": None,
                "signal_breakdown": None
            })
            continue

        resume_skills = set(processor.extract_skills(resume_text))

        # Score using the 3-signal engine
        result = engine.score(resume_text, jd_text, resume_skills, jd_skills)
        explanation = engine.explain(result["signals"])

        print(f"[SCORE] {resume_file.filename}: score={result['score']}, signals={result['signals']}")

        results.append({
            "filename": resume_file.filename,
            "score": result["score"],
            "skills": sorted(list(resume_skills)),
            "explanation": {
                "skill_overlap_impact": explanation["Skill Match Impact"],
                "cosine_similarity_impact": explanation["Text Similarity Impact"],
                "keyword_density_impact": explanation["Keyword Coverage Impact"],
            },
            "signal_breakdown": result["signals"]
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return {"ranked_resumes": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
