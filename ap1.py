import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

st.set_page_config(page_title="Next Word Predictor", page_icon="🇵🇰")

st.title("Next Word Prediction System")
st.markdown("---")
st.write("Pakistani Rupee History based NLP Model")

# 1. Model aur Tokenizer Load Karna (Directly)
@st.cache_resource
def load_model_objects():
    model = tf.keras.models.load_model('next_word_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

try:
    model, tokenizer = load_model_objects()
    max_sequence_len = 127 # Wahi jo training mein tha

    # 2. User Input
    input_text = st.text_input("Apna sentence likhein:", placeholder="Example: State Bank of")

    if st.button("Predict"):
        if input_text:
            # Tokenization aur Padding
            token_list = tokenizer.texts_to_sequences([input_text.lower()])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
            
            # Prediction logic
            predicted = model.predict(token_list, verbose=0)
            predicted_word_index = np.argmax(predicted, axis=1)[0]
            
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted_word_index:
                    output_word = word
                    break
            
            if output_word:
                st.success(f"Next Word: **{output_word}**")
                st.info(f"Sentence: {input_text} **{output_word}**")
            else:
                st.warning("Model ko iska agla word nahi pata.")
        else:
            st.warning("Pehle kuch type karein!")

except Exception as e:
    st.error(f"Model load karne mein masla aaya: {e}")