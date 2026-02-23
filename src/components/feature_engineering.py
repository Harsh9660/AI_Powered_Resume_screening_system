import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.processor import ResumeProcessor

class FeatureExtractor:
    def __init__(self):
        self.processor = ResumeProcessor()
        self.tfidf = TfidfVectorizer()

    def get_features(self, df):
        """
        Extracts features from the dataframe containing resume_text and job_description.
        """
        df['processed_resume'] = df['resume_text'].apply(lambda x: self.processor.process_text(x)['cleaned_text'])
        df['processed_jd'] = df['job_description'].apply(lambda x: self.processor.process_text(x)['cleaned_text'])
        
        def get_skill_overlap(row):
            resume_skills = set(self.processor.extract_skills(row['resume_text']))
            jd_skills = set(self.processor.extract_skills(row['job_description']))
            if not jd_skills:
                return 0.0
            return len(resume_skills.intersection(jd_skills)) / len(jd_skills)

        df['skill_overlap'] = df.apply(get_skill_overlap, axis=1)

        similarities = []
        for i, row in df.iterrows():
            sim = self.processor.get_similarity(row['processed_resume'], row['processed_jd'])
            similarities.append(sim)
        
        df['cosine_similarity'] = similarities
        
        return df

if __name__ == "__main__":
    df = pd.read_csv("data/raw/synthetic_data.csv")
    extractor = FeatureExtractor()
    df_features = extractor.get_features(df)
    df_features.to_csv("data/processed/featured_data.csv", index=False)
    print("Features extracted and saved to data/processed/featured_data.csv")
