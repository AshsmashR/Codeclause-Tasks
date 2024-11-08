# Import necessary libraries

import os
import re
import pandas as pd
import nltk
from docx import Document
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords')
nltk.download('punkt')
#loading my dataset and listing words for OCEAN traits
resume_directory = r"E:\Resumes"  # Folder containing .docx resumes
conscientiousness_words = ["organized", "managed", "responsible", "detail", "plan", "structured", "achieved", 
                           "implemented", "deadlines", "monitored", "supervised", "meticulous", "diligent", 
                           "punctual", "systematic", "efficient", "quality", "standards", "goals", "productivity", "deliverables"]

extraversion_words = ["enthusiastic", "motivated", "proactive", "leadership", "driven", "outgoing", "energetic", 
                      "dynamic", "inspired", "ambitious", "engaged", "influential", "collaborated", "networked", 
                      "led", "presented", "facilitated", "optimistic", "visionary"]

agreeableness_words = ["collaborated", "team", "supported", "assisted", "helped", "friendly", "cooperative", 
                       "empathetic", "partnership", "harmonious", "understanding", "volunteer", "mentoring", 
                       "engaged", "compassionate", "interpersonal", "relations", "accommodating", "encouraged"]

openness_words = ["innovative", "creative", "conceptualized", "visionary", "inventive", "ideas", "analysis", 
                  "curiosity", "exploration", "culture", "new approaches", "research", "forward-thinking", 
                  "intellectual", "strategic", "philosophy", "diverse", "insights", "inspired", "flexible"]

neuroticism_words = ["challenges", "difficult", "stress", "issues", "worried", "anxious", "problems", 
                     "struggle", "demanding", "pressure", "tense", "concerns", "crisis", "setbacks", 
                     "hurdles", "frustrated", "struggled", "complex", "risk", "complaints", "fatigue"]

stress_keywords = ["challenge", "struggle", "difficult", "issue", "problem", "stress", 
                   "pressure", "failure", "crisis", "concerns", "frustrated", "hurdles", 
                   "setback", "demanding", "complex", "risk", "complaints", "fatigue"]

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

def lexical_diversity(text):
    """Calculates lexical diversity as a measure for Openness."""
    words = word_tokenize(text)
    unique_words = set(word for word in words if word.isalpha())
    return len(unique_words) / len(words) if words else 0

def sentiment_score(text):
    """Calculates sentiment score, useful for Extraversion and Neuroticism."""
    return TextBlob(text).sentiment.polarity

def count_stress_words(text):
    """Counts stress-related words in text for Neuroticism."""
    words = word_tokenize(text.lower())
    return sum(1 for word in words if word in stress_keywords)

processed_texts = {}
for filename in os.listdir(resume_directory):
    if filename.endswith(".docx"):
        file_path = os.path.join(resume_directory, filename)
        text = extract_text_from_docx(file_path)
        processed_texts[filename] = preprocess_text(text)

#TF-IDF features to tokensize my words
vectorizer_tfidf = TfidfVectorizer(max_features=5000)
tfidf_features = vectorizer_tfidf.fit_transform(processed_texts.values())
tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=vectorizer_tfidf.get_feature_names_out(), index=processed_texts.keys())

#clustering for label generation for my dataset
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(tfidf_df)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(tfidf_df)
tfidf_df['Cluster'] = clusters

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
personality_scores = {filename: calculate_personality_scores(processed_texts[filename], tfidf_df.loc[filename, 'Cluster'])
                      for filename in processed_texts.keys()}
ocean_df = pd.DataFrame.from_dict(personality_scores, orient='index')
tfidf_df = pd.concat([tfidf_df, ocean_df], axis=1)

#the updated DataFrame with all five traits
print(tfidf_df[['Cluster', 'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']].head(10))

#
#labels according to my cluster assignmnets
print("Labels (Cluster Assignments) and Selected Features:")
tfidf_df_display = tfidf_df[['Cluster'] + list(tfidf_df.columns[:10])]  # Adjust the number of columns as needed
print(tfidf_df_display.head(10))
print("\nPersonality Traits with Labels:")
print(tfidf_df[['Cluster', 'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']].head(10))
labels = tfidf_df['Cluster'].tolist()
print("Labels list:", labels)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

#model for NLP
X = tfidf_df.drop(columns=['Cluster', 'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'])
y = tfidf_df['Cluster']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_y_pred = rf_classifier.predict(X_test)

print("Random Forest Classifier Performance:")
print(f"Accuracy: {accuracy_score(y_test, rf_y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, rf_y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_y_pred))

svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train_scaled, y_train)
svm_y_pred = svm_classifier.predict(X_test_scaled)
print("\nSupport Vector Machine Classifier Performance:")
print(f"Accuracy: {accuracy_score(y_test, svm_y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, svm_y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_y_pred))

# Import necessary libraries for saving/loading the model and making predictions on a single resume
import joblib

# Save models, vectorizer, and scaler for later use
joblib.dump(rf_classifier, 'rf_classifier_model.joblib')
#joblib.dump(svm_classifier, 'svm_classifier_model.joblib')
joblib.dump(vectorizer_tfidf, 'tfidf_vectorizer.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Define a function to calculate personality traits based on the predicted cluster
def predict_personality_for_single_resume(file_path, model, vectorizer, scaler=None):
    # Extract and preprocess the resume text
    text = extract_text_from_docx(file_path)
    processed_text = preprocess_text(text)
    
    # Transform text using the TF-IDF vectorizer
    tfidf_vector = vectorizer.transform([processed_text])
    
    # Convert to DataFrame with correct feature names
    tfidf_vector_df = pd.DataFrame(tfidf_vector.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Scale if SVM is used and scaler is provided
    if scaler:
        tfidf_vector_df = scaler.transform(tfidf_vector_df)  # Scaling will work correctly with DataFrame input
    
    # Predict the cluster
    predicted_cluster = model.predict(tfidf_vector_df)[0]
    
    # Calculate personality traits based on the predicted cluster
    traits = calculate_personality_traits(predicted_cluster) #type: ignore
    
    # Format the result in a DataFrame
    results_df = pd.DataFrame([{
        'Cluster': predicted_cluster,
        'Openness': traits['Openness'],
        'Conscientiousness': traits['Conscientiousness'],
        'Extraversion': traits['Extraversion'],
        'Agreeableness': traits['Agreeableness'],
        'Neuroticism': traits['Neuroticism']
    }], index=[os.path.basename(file_path)])
    
    return results_df
# Load the saved models and vectorizer if needed (for a real scenario, you'd load these from disk)
rf_classifier = joblib.load('rf_classifier_model.joblib')
#svm_classifier = joblib.load('svm_classifier_model.joblib')
vectorizer_tfidf = joblib.load('tfidf_vectorizer.joblib')
scaler = joblib.load('scaler.joblib')

# Predict personality traits for a single resume using Random Forest or SVM
single_resume_path = r"C:\Users\Paru\OneDrive\Documents\AISWARYA RANI.docx"  # Replace with actual path of the resume to test

# Uncomment the model you wish to use
# results_df = predict_personality_for_single_resume(single_resume_path, rf_classifier, vectorizer_tfidf)
results_df = predict_personality_for_single_resume(single_resume_path, rf_classifier, vectorizer_tfidf, scaler)

# Display the prediction in the desired format
print("Predicted Personality Traits for the Resume:")
print(results_df)

#The resume AISWARYA RANI.docx suggests a moderately conscientious individual (Conscientiousness: 0.7),
# balanced in openness, extraversion, agreeableness, and low in neuroticism (Cluster: 4)

# Define a function to calculate personality traits based on the predicted cluster
def predict_personality_for_single_resume(file_path, model, vectorizer, scaler=None):
    # Extract and preprocess the resume text
    text = extract_text_from_docx(file_path)
    processed_text = preprocess_text(text)
    
    # Transform text using the TF-IDF vectorizer
    tfidf_vector = vectorizer.transform([processed_text])
    
    # Convert to DataFrame with correct feature names
    tfidf_vector_df = pd.DataFrame(tfidf_vector.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Scale if SVM is used and scaler is provided
    if scaler:
        tfidf_vector_df = scaler.transform(tfidf_vector_df)  # Scaling will work correctly with DataFrame input
    
    # Predict the cluster
    predicted_cluster = model.predict(tfidf_vector_df)[0]
    
    # Calculate personality traits based on the predicted cluster
    traits = calculate_personality_scores(text, predicted_cluster)
    
    # Format the result in a DataFrame
    results_df = pd.DataFrame([{
        'Cluster': predicted_cluster,
        'Openness': traits['Openness'],
        'Conscientiousness': traits['Conscientiousness'],
        'Extraversion': traits['Extraversion'],
        'Agreeableness': traits['Agreeableness'],
        'Neuroticism': traits['Neuroticism']
    }], index=[os.path.basename(file_path)])
    
    return results_df

# Run the prediction for a single resume
results_df = predict_personality_for_single_resume(single_resume_path, rf_classifier, vectorizer_tfidf, scaler)
single_resume_path = r"C:\Users\Paru\OneDrive\Documents\AISWARYA RANI.docx"  # Replace with actual path of the resume to test

# Uncomment the model you wish to use
# results_df = predict_personality_for_single_resume(single_resume_path, rf_classifier, vectorizer_tfidf)
results_df = predict_personality_for_single_resume(single_resume_path, rf_classifier, vectorizer_tfidf, scaler)

# Display the prediction in the desired format
print("Predicted Personality Traits for the Resume:")
print(results_df)
