import os
import streamlit as st
from predict import load_model, top_predict, MODEL_PATH

st.title("ticket classifier")
st.write("paste a support ticket and get top 3 predicted ticket types.")

if not os.path.exists(MODEL_PATH):
    st.error("model not found, run : python train.py to generate models/model.joblib")
    st.stop()

@st.cache_resource
def get_model():
    return load_model

model = load_model()

text = st.text_area("ticket text", height=200)

if st.button("predict"):
    if not text.strip():
        st.warning("please enter some text.")
    else:
        preds = top_predict(model, text)
        st.subheader("top 3 predictions")
        for p in preds:
            st.write(f"- {p['label']}: {p['proba']*100:.1f}%")
