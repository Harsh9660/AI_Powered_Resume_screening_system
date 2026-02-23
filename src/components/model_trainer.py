import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error
import joblib
import os

class ModelTrainer:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.X = self.df[['skill_overlap', 'cosine_similarity']]
        self.y = self.df['relevance_score']
        self.y_binary = self.df['label']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        _, _, self.y_train_binary, self.y_test_binary = train_test_split(
            self.X, self.y_binary, test_size=0.2, random_state=42
        )

    def objective(self, trial):
        param = {
            'verbosity': 0,
            'objective': 'reg:squarederror',
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'eta': trial.suggest_float('eta', 1e-8, 1.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        }
        
        model = xgb.XGBRegressor(**param)
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, preds)
        return mse

    def tune(self, n_trials=50):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        print(f"Best parameters: {study.best_params}")
        return study.best_params

    def train_final(self, best_params):
        model = xgb.XGBRegressor(**best_params)
        model.fit(self.X_train, self.y_train)
        
        # Save model
        os.makedirs("src/models", exist_ok=True)
        joblib.dump(model, "src/models/resume_ranker.pkl")
        print("Model saved to src/models/resume_ranker.pkl")
        
        # Evaluation
        preds = model.predict(self.X_test)
        binary_preds = [1 if p > 0.8 else 0 for p in preds]
        f1 = f1_score(self.y_test_binary, binary_preds)
        print(f"Evaluation F1 Score: {f1}")
        
        return model

if __name__ == "__main__":
    trainer = ModelTrainer("data/processed/featured_data.csv")
    best_params = trainer.tune(n_trials=20)
    trainer.train_final(best_params)
