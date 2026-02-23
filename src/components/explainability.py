import shap
import joblib
import pandas as pd
import os

class Explainer:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        try:
            # Try TreeExplainer first for speed and accuracy
            self.explainer = shap.TreeExplainer(self.model.get_booster())
        except Exception as e:
            print(f"TreeExplainer failed: {e}. Falling back to basic Explainer.")
            try:
                self.explainer = shap.Explainer(self.model)
            except:
                self.explainer = None

    def explain(self, features_df):
        """
        Calculates SHAP values for the given features dataframe.
        """
        if self.explainer is None:
            return None
        shap_values = self.explainer(features_df)
        return shap_values

if __name__ == "__main__":
    
    model_path = "src/models/resume_ranker.pkl"
    if os.path.exists(model_path):
        explainer = Explainer(model_path)
        sample_data = pd.DataFrame([[0.8, 0.7]], columns=['skill_overlap', 'cosine_similarity'])
        shap_vals = explainer.explain(sample_data)
        print("SHAP explanation generated.")
        print(shap_vals)
    else:
        print("Model file not found. Run model_trainer.py first.")
