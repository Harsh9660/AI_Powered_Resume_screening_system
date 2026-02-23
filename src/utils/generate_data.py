import pandas as pd
import random
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_md")

skills_pool = [
    "Python", "Java", "C++", "C#", "JavaScript", "TypeScript", "HTML", "CSS", "React", "Angular", "Vue", 
    "Node.js", "Express", "Django", "Flask", "FastAPI", "Machine Learning", "Deep Learning", "NLP", 
    "Computer Vision", "Data Science", "Data Analysis", "SQL", "NoSQL", "MongoDB", "PostgreSQL", 
    "MySQL", "Docker", "Kubernetes", "AWS", "Azure", "GCP", "CI/CD", "Git", "GitHub", "Linux", 
    "REST API", "GraphQL", "TensorFlow", "PyTorch", "Keras", "Scikit-Learn", "Pandas", "NumPy", 
    "Spark", "Hadoop", "Tableau", "Power BI", "Excel", "Project Management", "Agile", "Scrum", 
    "Communication Skills", "Problem Solving", "Cloud Computing", "Microservices", "Jenkins", 
    "Terraform", "Ansible", "Redis", "Elasticsearch", "Selenium", "DevOps"
]
roles = ["Data Scientist", "Software Engineer", "DevOps Engineer", "Frontend Developer", "ML Engineer"]

def generate_resume_text(role):
    skills = random.sample(skills_pool, k=random.randint(3, 8))
    experience = f"{random.randint(1, 15)} years of experience in {role}."
    education = random.choice(["BS in Computer Science", "MS in Data Science", "PhD in AI", "Self-taught Developer"])
    return f"Role: {role}. Experience: {experience}. Education: {education}. Skills: {', '.join(skills)}."

def generate_jd(role):
    skills = random.sample(skills_pool, k=random.randint(4, 7))
    return f"Looking for a {role} with expertise in {', '.join(skills)}."

def get_cleaned_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])

data = []
print("Generating 500 synthetic data points...")
for i in range(500):
    role = random.choice(roles)
    resume_text = generate_resume_text(role)
    
    jd_role = random.choice(roles)
    jd_text = generate_jd(jd_role)
    
    # Calculate features for ground truth
    res_skills = set([s for s in skills_pool if s.lower() in resume_text.lower()])
    jd_skills = set([s for s in skills_pool if s.lower() in jd_text.lower()])
    
    overlap = len(res_skills.intersection(jd_skills)) / len(jd_skills) if jd_skills else 0
    
    # Simple semantic similarity proxy with more variance
    similarity_proxy = random.uniform(0.1, 0.9)
    if role == jd_role:
        similarity_proxy = random.uniform(0.6, 1.0)
    
    # Target score: with significant noise to prevent model bias to a single value
    ground_truth_score = (0.5 * overlap) + (0.4 * similarity_proxy) + random.uniform(-0.05, 0.05)
    ground_truth_score = max(0.0, min(ground_truth_score, 1.0))
    
    data.append({
        "resume_id": i,
        "resume_text": resume_text,
        "job_description": jd_text,
        "relevance_score": ground_truth_score,
        "label": 1 if ground_truth_score > 0.65 else 0
    })

df = pd.DataFrame(data)
df.to_csv("data/raw/synthetic_data.csv", index=False)
print("Improved synthetic dataset created at data/raw/synthetic_data.csv")
