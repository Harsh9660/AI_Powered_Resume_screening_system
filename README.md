# AI-Powered Resume Screening System (Industry-Level)

This project automates the resume filtering process using NLP and Machine Learning. It parses resumes, computes matching scores against job descriptions, and ranks them using an XGBoost model.

## 🎯 Features
- **Resume Parser**: PDF to structured text extraction.
- **NLP Matching**: Skill extraction and semantic similarity scores.
- **Ranking Model**: XGBoost trained with Optuna hyperparameter tuning.
- **Explainability**: SHAP integration to explain ranking scores.
- **FastAPI**: Real-time ranking via API endpoints.

## 🚀 Getting Started

### Installation
```bash
pip install -r requirements.txt
python3 -m spacy download en_core_web_md
```

### Run the API
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 src/api/main.py
```

### Usage
Upload multiple PDF resumes and provide a Job Description (JD) text to the `/rank-resumes` endpoint.

## 📂 Project Structure
- `src/api/`: FastAPI implementation.
- `src/components/`: Feature engineering and model training.
- `src/utils/`: PDF parsing and NLP processing utilities.
- `data/`: Raw and processed datasets.
- `tests/`: Automated test scripts.

