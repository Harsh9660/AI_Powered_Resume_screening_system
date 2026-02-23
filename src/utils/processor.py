import spacy
import re
from spacy.matcher import PhraseMatcher

# Common tech abbreviations → full names that exist in skills list
ABBREV_MAP = {
    r'\bML\b': 'Machine Learning',
    r'\bDL\b': 'Deep Learning',
    r'\bNLP\b': 'NLP',
    r'\bCV\b': 'Computer Vision',
    r'\bDS\b': 'Data Science',
    r'\bDA\b': 'Data Analysis',
    r'\bAI\b': 'Machine Learning',
    r'\bPG\b': 'PostgreSQL',
    r'\bK8s\b': 'Kubernetes',
    r'\bJS\b': 'JavaScript',
    r'\bTS\b': 'TypeScript',
    r'\bTF\b': 'TensorFlow',
    r'\bPT\b': 'PyTorch',
    r'\bGH\b': 'GitHub',
    r'\bCI/CD\b': 'CI/CD',
}

class ResumeProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
        self.skills_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.skills_list = [
            "Python", "Java", "C++", "C#", "JavaScript", "TypeScript", "HTML", "CSS", "React", "Angular", "Vue", 
            "Node.js", "Express", "Django", "Flask", "FastAPI", "Machine Learning", "Deep Learning", "NLP", 
            "Computer Vision", "Data Science", "Data Analysis", "SQL", "NoSQL", "MongoDB", "PostgreSQL", 
            "MySQL", "Docker", "Kubernetes", "AWS", "Azure", "GCP", "CI/CD", "Git", "GitHub", "Linux", 
            "REST API", "GraphQL", "TensorFlow", "PyTorch", "Keras", "Scikit-Learn", "Pandas", "NumPy", 
            "Spark", "Hadoop", "Tableau", "Power BI", "Excel", "Project Management", "Agile", "Scrum", 
            "Communication Skills", "Problem Solving", "Cloud Computing", "Microservices", "Jenkins", 
            "Terraform", "Ansible", "Redis", "Elasticsearch", "Selenium", "DevOps"
        ]
        patterns = [self.nlp.make_doc(skill.lower()) for skill in self.skills_list]
        self.skills_matcher.add("SKILLS", patterns)

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common tech abbreviations to full terms."""
        for pattern, replacement in ABBREV_MAP.items():
            text = re.sub(pattern, replacement, text)
        return text

    def extract_skills(self, text):
        expanded = self._expand_abbreviations(text)
        doc = self.nlp(expanded.lower())
        matches = self.skills_matcher(doc)
        skills = set()
        for match_id, start, end in matches:
            # Return canonical (capitalized) skill name
            matched_text = doc[start:end].text.title()
            # Find exact match in skills list (case-insensitive)
            for skill in self.skills_list:
                if skill.lower() == matched_text.lower() or skill.lower() == doc[start:end].text.lower():
                    skills.add(skill)
                    break
            else:
                skills.add(matched_text)
        return list(skills)

    def get_similarity(self, text1, text2):
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        return float(doc1.similarity(doc2))

    def process_text(self, text):
        """
        Cleans and processes text, returns basic info.
        """
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        skills = self.extract_skills(text)
        return {
            "entities": entities,
            "skills": skills,
            "cleaned_text": " ".join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])
        }

if __name__ == "__main__":
    processor = ResumeProcessor()
    text = "Hi, I am a Data Scientist with 5 years of experience in Python and Machine Learning. I have worked with Docker and AWS."
    print(processor.process_text(text))
