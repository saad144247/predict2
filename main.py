from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

app = FastAPI()

# Model aur Tokenizer load karna
model = tf.keras.models.load_model('next_word_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Training ke waqt jo max_length thi (Humein prediction ke liye chahiye)
# Agar aapne mera pichla code use kiya hai toh ye wahi length hogi
max_sequence_len = 127 # Ye aapki training file ke hisaab se auto-adjust hota hai

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict_next_word(data: InputText):
    token_list = tokenizer.texts_to_sequences([data.text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            output_word = word
            break
    
    return {"input": data.text, "next_word": output_word}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)