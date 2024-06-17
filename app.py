import streamlit as st
import pickle
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stemmer = PorterStemmer()

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)
    new_text = []
    for i in text:
        if i.isalnum() and i not in stop_words and i not in string.punctuation:
            new_text.append(stemmer.stem(i))

    return " ".join(new_text)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.title("Email/ Spam Classifier")


input_sms = st.text_area("Enter the message")

if st.button('predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
