"""
Scoring Engine - Replaces the XGBoost black-box model with a transparent,
multi-signal formula that produces genuinely varied scores per resume.

Score = 0.40 * skill_overlap + 0.35 * tfidf_similarity + 0.25 * keyword_density
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class ScoringEngine:
    def __init__(self):
        self.weights = {
            "skill_overlap": 0.40,
            "tfidf_similarity": 0.35,
            "keyword_density": 0.25,
        }

    def _compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """TF-IDF cosine similarity — much more sensitive than spaCy vectors."""
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),       # capture bigrams like "machine learning"
                sublinear_tf=True,        # dampen high-frequency terms
                min_df=1
            )
            matrix = vectorizer.fit_transform([text1, text2])
            sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
            return float(sim)
        except Exception:
            return 0.0

    def _compute_keyword_density(self, resume_text: str, jd_text: str) -> float:
        """
        Fraction of unique JD words that appear in the resume text.
        Case-insensitive. Ignores very short stop words.
        """
        # Extract meaningful words from JD (len > 3, only alphanumeric)
        jd_words = set(
            w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', jd_text)
        )
        if not jd_words:
            return 0.0
        resume_lower = resume_text.lower()
        matched = sum(1 for w in jd_words if w in resume_lower)
        return matched / len(jd_words)

    def score(self, resume_text: str, jd_text: str,
              resume_skills: set, jd_skills: set) -> dict:
        """
        Returns the final score (0-1) and individual signal values.
        """
        # Signal 1: Skill overlap
        skill_overlap = (
            len(resume_skills.intersection(jd_skills)) / len(jd_skills)
            if jd_skills else 0.0
        )

        # Signal 2: TF-IDF similarity
        tfidf_sim = self._compute_tfidf_similarity(resume_text, jd_text)

        # Signal 3: Keyword density
        kw_density = self._compute_keyword_density(resume_text, jd_text)

        # Weighted final score
        final = (
            self.weights["skill_overlap"] * skill_overlap
            + self.weights["tfidf_similarity"] * tfidf_sim
            + self.weights["keyword_density"] * kw_density
        )
        final = round(min(max(final, 0.0), 1.0), 4)

        return {
            "score": final,
            "signals": {
                "skill_overlap": round(skill_overlap, 4),
                "tfidf_similarity": round(tfidf_sim, 4),
                "keyword_density": round(kw_density, 4),
            }
        }

    def explain(self, signals: dict) -> dict:
        """
        Returns the weighted contribution of each signal to the final score.
        This replaces SHAP with direct, transparent attribution.
        """
        return {
            "Skill Match Impact": round(signals["skill_overlap"] * self.weights["skill_overlap"], 4),
            "Text Similarity Impact": round(signals["tfidf_similarity"] * self.weights["tfidf_similarity"], 4),
            "Keyword Coverage Impact": round(signals["keyword_density"] * self.weights["keyword_density"], 4),
        }
