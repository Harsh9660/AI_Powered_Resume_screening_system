import os
import joblib
import logging
from typing import Dict, Any

import pandas as pd
from src.utils.processor import ResumeProcessor

# ---------------------------------------
# Logging Configuration
# ---------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------
# Resume Ranking System
# ---------------------------------------
class ResumeRanker:
    def __init__(self, model_path: str):
        self.processor = ResumeProcessor()
        self.model = self._load_model(model_path)

    def _load_model(self, path: str):
        """Safely load ML model."""
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                logger.info("Model loaded successfully.")
                return model
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return None
        else:
            logger.warning("Model file not found. Using rule-based scoring.")
            return None

    def evaluate(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """Evaluate resume against job description."""
        try:
            # Process texts
            res_data = self.processor.process_text(resume_text)
            jd_data = self.processor.process_text(jd_text)

            res_skills = set(res_data.get("skills", []))
            jd_skills = set(jd_data.get("skills", []))

            # Skill overlap
            overlap = (
                len(res_skills.intersection(jd_skills)) / len(jd_skills)
                if jd_skills else 0.0
            )

            # Semantic similarity
            similarity = self.processor.get_similarity(
                res_data.get("cleaned_text", ""),
                jd_data.get("cleaned_text", "")
            )

            # Feature dataframe
            features = pd.DataFrame(
                [[overlap, similarity]],
                columns=["skill_overlap", "cosine_similarity"]
            )

            # Model prediction or fallback scoring
            if self.model:
                score = float(self.model.predict(features)[0])
            else:
                # Weighted rule-based fallback
                score = round((0.6 * overlap) + (0.4 * similarity), 4)

            return {
                "skills_found": list(res_skills),
                "jd_skills": list(jd_skills),
                "skill_overlap": round(overlap, 4),
                "similarity": round(similarity, 4),
                "final_score": round(score, 4)
            }

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "error": str(e)
            }


# ---------------------------------------
# Example Usage
# ---------------------------------------
if __name__ == "__main__":

    MODEL_PATH = "src/models/resume_ranker.pkl"

    jd_text = """
    Familiarity with REST APIs and model deployment.
    Strong problem-solving skills.
    Experience with cloud platforms (AWS/GCP/Azure).
    Knowledge of Docker & CI/CD.
    Experience working on end-to-end ML projects.
    """

    resumes = {
        "ML/Backend Candidate": 
        "Experienced Python developer with AWS, Docker, CI/CD, REST APIs, and ML deployment experience.",
        
        "Frontend Candidate":
        "Frontend developer skilled in React, CSS, JavaScript, and UI/UX design."
    }

    ranker = ResumeRanker(MODEL_PATH)

    for name, text in resumes.items():
        result = ranker.evaluate(text, jd_text)

        print(f"\n===== {name} =====")
        if "error" in result:
            print("Error:", result["error"])
        else:
            print("Skills Found:", result["skills_found"])
            print("JD Skills:", result["jd_skills"])
            print("Skill Overlap:", result["skill_overlap"])
            print("Semantic Similarity:", result["similarity"])
            print("Final Score:", result["final_score"])