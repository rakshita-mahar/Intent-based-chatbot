import os
import random
import nltk
import json
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import streamlit as st

intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "How are you?", "Hey"],
            "responses": ["Hello!", "Hi there!", "Greetings!"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you", "Goodbye"],
            "responses": ["Goodbye!", "See you later!", "Take care!"]
        },
        {
            "tag": "recommend_book",
            "patterns": ["Suggest me a book", "I want a book recommendation", "Can you recommend a book?"],
            "responses": ["Sure! What genre do you prefer?", "Do you like fiction or non-fiction?", "Tell me your favorite genre."]
        },
        {
            "tag": "book_suggestion_fiction",
            "patterns": ["I like fiction", "Recommend me a fiction book"],
            "responses": ["I recommend 'The Great Gatsby'.", "How about '1984' by George Orwell?"]
        },
        {
            "tag": "book_suggestion_non_fiction",
            "patterns": ["I like non-fiction", "Recommend me a non-fiction book"],
            "responses": ["You might enjoy 'Sapiens: A Brief History of Humankind' by Yuval Noah Harari.", "How about 'Educated' by Tara Westover?"]
        }
    ]
}

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def preprocess_data(intents):
    training_sentences = []
    training_labels = []
    class_labels = []
    response_dict = {}
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern.lower())
            training_sentences.append(" ".join([lemmatizer.lemmatize(w) for w in word_list]))
            training_labels.append(intent['tag'])
        
        if intent['tag'] not in class_labels:
            class_labels.append(intent['tag'])
        
        response_dict[intent['tag']] = intent['responses']
    
    return training_sentences, training_labels, class_labels, response_dict

training_sentences, training_labels, class_labels, response_dict = preprocess_data(intents)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(training_sentences).toarray()
encoder = LabelEncoder()
y_train = encoder.fit_transform(training_labels)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

def predict_intent(text):
    text_vector = vectorizer.transform([text]).toarray()
    prediction = model.predict(text_vector)
    predicted_intent = encoder.inverse_transform(prediction)[0]
    return predicted_intent

st.title("Book Recommendation Chatbot")
st.write("Welcome to the Book Recommendation Chatbot! Ask me about book recommendations.")

user_input = st.text_input("You: ", "")

if user_input:
    intent = predict_intent(user_input)
    response = random.choice(response_dict[intent])
    st.write(f"Bot: {response}")

