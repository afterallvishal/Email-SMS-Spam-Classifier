import streamlit as st
import pickle
import string
import nltk


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')   

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer



#load the vectorizer and model
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

#Initialise stemmer
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))   



#preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

#Steamlit UI
st.title('Email/SMS Spam Classifier')
input_sms=st.text_input("Enter the Message")

if st.button('Predict'):
    #preprocess
    transformed_sms=transform_text(input_sms)

    #vectorize
    vector_input=tfidf.transform([transformed_sms])

    #Predict
    result=model.predict(vector_input)[0]

    if result==1:
        st.header('Spam!')
    else:
        st.header('Not a Spam!')

