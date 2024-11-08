import gradio as gr
import joblib
import pandas as pd
import re
from docx import Document
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import nltk

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Load pre-trained models and vectorizers
rf_classifier = joblib.load('rf_classifier_model.joblib')
vectorizer_tfidf = joblib.load('tfidf_vectorizer.joblib')
scaler = joblib.load('scaler.joblib')

# Define functions

def extract_text_from_docx(file_path):
    """Extracts text from a .docx file."""
    doc = Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])

def preprocess_text(text):
    """Removes punctuation, numbers, stopwords, and converts text to lowercase."""
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    words = word_tokenize(text.lower())
    return ' '.join([word for word in words if word.isalpha() and word not in stopwords.words('english')])

def sentiment_score(text):
    """Calculates sentiment score, useful for Extraversion and Neuroticism."""
    return TextBlob(text).sentiment.polarity

def count_stress_words(text):
    """Counts stress-related words in text for Neuroticism."""
    stress_keywords = ["challenge", "struggle", "difficult", "issue", "problem", "stress", 
                       "pressure", "failure", "crisis", "concerns", "frustrated", "hurdles", 
                       "setback", "demanding", "complex", "risk", "complaints", "fatigue"]
    words = word_tokenize(text.lower())
    return sum(1 for word in words if word in stress_keywords)

def calculate_personality_scores(text, cluster):
    """Maps personality traits based on text and cluster characteristics."""
    scores = {}
    sentiment = sentiment_score(text)
    stress_count = count_stress_words(text)
    neuroticism = min(1.0, 0.7 if cluster == 0 else 0.4 if cluster == 1 else 0.2 if sentiment < 0 else 0) + min(0.1 * stress_count, 1.0)

    scores['Openness'] = 1.0 if cluster == 2 else 0.5
    scores['Conscientiousness'] = 1.0 if cluster == 0 else 0.8 if cluster == 1 else 0.7
    scores['Extraversion'] = max(0, sentiment) + (0.7 if cluster == 2 else 0.3)
    scores['Agreeableness'] = 0.9 if cluster == 1 else 0.5
    scores['Neuroticism'] = neuroticism

    return scores

# Gradio function to predict personality traits from uploaded resume
def predict_personality_for_uploaded_resume(file):
    # Extract and preprocess resume text
    text = extract_text_from_docx(file.name)
    processed_text = preprocess_text(text)
    
    # Transform text using TF-IDF and scale
    tfidf_vector = vectorizer_tfidf.transform([processed_text])
    tfidf_vector_df = pd.DataFrame(tfidf_vector.toarray(), columns=vectorizer_tfidf.get_feature_names_out())
    tfidf_vector_df = scaler.transform(tfidf_vector_df)
    
    # Predict personality cluster
    predicted_cluster = rf_classifier.predict(tfidf_vector_df)[0]
    traits = calculate_personality_scores(processed_text, predicted_cluster)
    
    return traits

# Create Gradio interface
iface = gr.Interface(
    fn=predict_personality_for_uploaded_resume, 
    inputs="file", 
    outputs="json",
    title="Personality Prediction from Resume",
    description="Upload a .docx resume to predict personality traits based on the OCEAN model."
)

iface.launch()
