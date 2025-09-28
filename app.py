from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
with open('models/model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('models/vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        job_title = request.form['title']
        job_description = request.form['description']
        job_requirements = request.form['requirements']
        
        # Combine job description and requirements
        job_data = job_description + " " + job_requirements
        
        # Vectorize the input data
        input_tfidf = vectorizer.transform([job_data])
        
        # Make prediction
        prediction = rf_model.predict(input_tfidf)
        
        # Display result
        result = 'Real Job Posting' if prediction == 0 else 'Fake Job Posting'
        
        return render_template('result.html', result=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
