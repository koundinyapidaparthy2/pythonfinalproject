import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Data Preprocessing
# Load the dataset
df = pd.read_csv("Sentiment_Analysis.csv", encoding="latin1", header=None, names=['target', 'ids', 'date', 'flag', 'user', 'text'])

# Drop unnecessary columns
df.drop(['ids', 'date', 'flag', 'user'], axis=1, inplace=True)

# Step 2: Feature Extraction
# Convert target values to integers
df['target'] = df['target'].replace({0: 'negative', 2: 'neutral', 4: 'positive'})

# Splitting into features and target
X = df['text']
y = df['target']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Training Naïve Bayes Classifier
# Create Bag of Words representation
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

# Convert to TF-IDF representation
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Train Naïve Bayes classifier
clf = MultinomialNB().fit(X_train_tfidf, y_train)

# Step 4: Evaluation
# Transform test set into TF-IDF representation
X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Predictions
y_pred = clf.predict(X_test_tfidf)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
