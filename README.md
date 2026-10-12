# SMS Spam Detector

**AICTE | IBM SkillsBuild — Machine Learning & Applied AI Internship 2026**

A machine learning web application that classifies SMS messages as Spam or Ham (legitimate) using Natural Language Processing and a Random Forest classifier.

---

## Project Overview

This project demonstrates end-to-end machine learning: data preprocessing, feature extraction, model training, evaluation, and deployment as a web app.

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest (100 trees) |
| Feature Extraction | TF-IDF Vectorizer (5000 character 2–5-gram features) |
| Dataset | UCI SMS Spam Collection (5,572 messages) |
| Accuracy | **98.57%** |
| Precision (Spam) | 100% |
| Recall (Spam) | 89% |

---

## How It Works

1. **Input**: User types any SMS/email message
2. **Preprocessing**: Text is vectorized using TF-IDF (Term Frequency–Inverse Document Frequency)
3. **Prediction**: Random Forest model classifies as Spam or Ham
4. **Output**: Result shown with confidence percentage

---

## Tech Stack

- Python 3.14 (pinned for deployment via `.python-version`)
- scikit-learn (ML model + TF-IDF)
- Flask (web server)
- HTML/CSS/JavaScript (frontend)
- pandas, numpy, joblib

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (generates model.pkl and vectorizer.pkl)
python train_model.py

# 3. Run the web app
python app.py

# 4. Open browser
http://localhost:5000
```

Run the smoke test with `python test_app.py`.

## Deployment (Render)

`render.yaml` builds with `pip install -r requirements.txt && python train_model.py` and starts with `gunicorn app:app`.
`.python-version` pins Python 3.14 so every dependency installs from a prebuilt wheel.

---

## Project Structure

```
spam-detector/
├── app.py              # Flask web server
├── train_model.py      # Model training script
├── spam.csv            # Dataset (UCI SMS Spam Collection)
├── model.pkl           # Trained Random Forest model
├── vectorizer.pkl      # TF-IDF vectorizer
├── requirements.txt    # Python dependencies
├── test_app.py         # Smoke test for the API
├── render.yaml         # Render deployment config
├── .python-version     # Python version pin for Render
├── README.md           # This file
└── templates/
    └── index.html      # Web interface
```

---

## Dataset

- **Source**: UCI Machine Learning Repository — SMS Spam Collection
- **Size**: 5,572 messages (4,825 ham + 747 spam)
- **Labels**: `ham` (legitimate), `spam`

---

## Features Used

- TF-IDF weighted character n-grams (2–5 chars, within word boundaries) — robust to SMS slang, typos and obfuscation like "fr33"
- Top 5,000 most frequent features kept, sublinear term frequency
- Balanced class weights so the minority spam class isn't under-learned

---

*Built as part of AICTE | IBM SkillsBuild ML & Applied AI Internship Program 2026*
