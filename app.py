from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import numpy as np
import pickle

st.title('Next Word Predictor')

model = load_model('nwp.keras')

with open('tokenizer.pkl','rb') as f:
    tokenizer = pickle.load(f)

ip = st.text_input('Enter Any Text:')
limit = st.number_input('Enter the no of future prediction', min_value=1, max_value=10, value=3)

btn = st.button('Predict Next Words')

def predict(text,limit):
    for i in range(limit):
      # tokenize
      token_text = tokenizer.texts_to_sequences([text])[0]

      # padding
      pad_token_text = pad_sequences([token_text],maxlen=17,padding='pre')

      # predict
      pos = np.argmax(model.predict(pad_token_text))

      for word,index in tokenizer.word_index.items():
        if index == pos:
          text = text + ' ' + word
    return text

if btn:
    st.write(predict(ip, limit))