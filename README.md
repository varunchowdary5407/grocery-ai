# 🛒 AI-Based Grocery Optimization System

Type a grocery list in plain English → **NLP** understands items & quantities → an **optimizer** finds the cheapest store plan.
The whole thing is wrapped in an **MLOps** pipeline.

## Architecture
1. **NLP** (`src/nlp.py`): text splitting, quantity/unit parsing, typo-tolerant product matching (TF-IDF character n-grams + cosine similarity).
2. **Optimization** (`src/optimize.py`): compares "one store" vs "cheapest item per store" including delivery fees.
3. **MLOps**
   - Experiment tracking & model selection: **MLflow** (`src/train.py`)
   - Model versioning + metrics: `models/metrics.json`
   - CI/CD: **GitHub Actions** retrains and tests on every push (`.github/workflows/ci.yml`)
   - Deployment: **Streamlit Community Cloud** auto-deploys from GitHub
   - Monitoring: request log + unrecognised-item rate inside the web app
4. **Website**: `app.py` (Streamlit) with Optimizer, MLOps and Monitoring tabs.

## Run locally / in Colab
```
pip install -r requirements-dev.txt
python -m src.make_data
python -m src.train
pytest -q
streamlit run app.py
```
