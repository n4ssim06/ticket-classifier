import joblib
import numpy as np

MODEL_PATH = "models/model.joblib"

def load_model():
    return joblib.load(MODEL_PATH)

def top_predict(model, text):
    proba = model.predict_proba([text])
    classes = model.classes_
    top = np.argsort(proba)[::-1][:3] # List of the 3 best proba indices from highest to lowest

    results = []
    for i in top:
        results.append({
            "label" : classes[i],
            "proba" : proba[i]
        })
    return results