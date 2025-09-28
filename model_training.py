import csv
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset (no pandas)
def load_data(file_path):
    X = []
    y = []
    with open(file_path, 'r', encoding='utf-8') as file:  # Added encoding='utf-8'
        reader = csv.DictReader(file)
        for row in reader:
            job_description = row['description'] + " " + row['requirements']  # Combining description and requirements
            X.append(job_description)
            y.append(int(row['fraudulent']))  # Fraudulent label (0 or 1)
    return X, y

# Preprocess the data
X, y = load_data('data/fake_job_postings.csv')

# Convert text data into numerical format using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the trained model and vectorizer
with open('models/model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

with open('models/vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Model and vectorizer saved successfully!")
