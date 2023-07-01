import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from streamlit_option_menu import option_menu
import streamlit as st
import re



class Predict_pipeline():
    def __init__(self, method='both') -> None:
        nltk.download('stopwords')
        self.stopwords = set(stopwords.words('english'))
        if method == 'both':
            self.bidirectional_lstm_model = load_model('YOUR_MODEL_PATH')
        elif method == 'title':
            self.bidirectional_lstm_model = load_model('YOUR_MODEL_PATH')

    def preprocess_data(self, data, vocab_size=5000, sent_length=20):
        '''Data Preprocess a list of words into model's inputs'''
        ps = PorterStemmer()
        corpus = []
        review = re.sub('[^a-zA-Z0-9]', ' ', data[0])  # Removing Special Characters
        review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if word.lower() not in self.stopwords]
        review = ' '.join(review)
        corpus.append(review)

        encoded_corpus = [one_hot(corpus[0], vocab_size)]
        padded_docs  = pad_sequences(sequences=encoded_corpus, maxlen=sent_length, padding='pre')
        X_final = np.array(padded_docs)
        return X_final

    def predict(self, preprocessed_data):
        y_prob = self.bidirectional_lstm_model.predict(preprocessed_data)
        y_pred = (y_prob > 0.5).astype(int)
        return y_pred


# Streamlit app

with st.sidebar:
    selected = option_menu(
        menu_title='Fake News Detective',
        menu_icon='🕵️',
        options=['News Title & Body', 'News Title only']
    )

if selected == 'News Title & Body':
    st.title('Fake News Detective 🕵️')
    st.text('Bidirectional LSTM Nueral Net to predict whether a given News is FAKE OR REAL !!')
    news_title = st.text_input('News Title ...')
    news_content = st.text_area('Main Content ...')
    predict_button = st.button('Predict')
    if predict_button:
        news_input = [f'{news_title}   {news_content}']
        pipeline = Predict_pipeline(method='both')
        preprocessed_input = pipeline.preprocess_data(data=news_input)
        preds = pipeline.predict(preprocessed_data=preprocessed_input)
        if preds[0][0] == 1:
            st.success('The News is REAL !! 😎')
            st.balloons()
        else: 
            st.text('News is FAKE 😢')

elif selected == 'News Title only':
    st.title('Fake News Detective 🕵️ : Title Only')
    st.text('Bidirectional LSTM Nueral Net to predict whether a given News is FAKE OR REAL !!')
    title = [st.text_input('News Title ...')]
    predict_button = st.button('Predict')
    if predict_button:
        pipeline = Predict_pipeline(method='title')
        preprocessed_input = pipeline.preprocess_data(data=title)
        preds = pipeline.predict(preprocessed_data=preprocessed_input)
        if preds[0][0] == 1:
            st.success('The News is REAL !! 😎')
            st.balloons()
        else: 
            st.text('News is FAKE 😢')