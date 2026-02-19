# ticket-classifier

Small project that classifies customer support tickets into 4 types:

- Incident
- Request
- Porblem
- Change

It trains a simple text model (TF-IDF + Logistic Regression) and provides a Streamlit app that shows the top 3 predicted classes with confidence scores.

## Dataset

This project uses the Hugging Face dataset: `Tobi-Bueck customer-support-tickets` (split: `train`).

## What the app does

1. You paste a support ticket text
2. The app loads a trained model from 'models/model.joblib'.
3. It returns:
    - the top 3 predicted ticket types
    - a confidence percentage for each prediction

## Requirements

- Python 3.10+ (works with 3.12)
- pip

## Setup

```bash
python 3 -m venv .venv
source .venv/bin.activate
pip install -r requirements.txt
```

## Train the model

This downloads the dataset (first run), trains the baseline model, pints metrics, and saves the model to `models/model.joblib`.

```bash
python train.py
```

## Run the app

```bash
streamlit run app.py
```
if you see "model not found", train it first with `python train.py`.

## project structure

- train.py - load data, filter English, train + evaluate, save the model (joblib)
- predict.py - load model and compute top 3 predictions
- app.py - Streamlit UI (paste text --> top 3 predictions)
- models/ - saved model file (`model.joblib`, generated)
- reports/ - optional outputs (confusion matrix, metrics)

## Notes

- This is a baseline intended to be simple and easy to understand.
- the dataset labels can be close in meaning (especially 'Problem' vs 'Incident'), so some confusion 