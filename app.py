import streamlit as st
import requests

st.title("Next Word Predictor 🇵🇰")
st.write("Pakistani Rupee History based NLP Model")

# User input
input_text = st.text_input("Kuch likhein (e.g. 'State Bank of'):", "")

if st.button("Predict Next Word"):
    if input_text:
        # FastAPI server ko data bhejna
        response = requests.post("http://127.0.0.1:8000/predict", json={"text": input_text})
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Word: **{result['next_word']}**")
        else:
            st.error("Backend se connect nahi ho saka.")
    else:
        st.warning("Pehlay kuch likhen!")