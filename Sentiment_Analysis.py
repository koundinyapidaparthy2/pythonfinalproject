import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
from sklearn.utils.multiclass import unique_labels

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

# Get unique labels from y_test and y_pred
labels = unique_labels(y_test, y_pred)

# Plot Confusion Matrix with dynamically determined display labels
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=labels)
disp.plot()
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot Learning Curves
train_sizes, train_scores, test_scores = learning_curve(clf, X_train_tfidf, y_train, cv=5, train_sizes=np.linspace(.1, 1.0, 5))
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='b', label='Cross-validation score')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='b')
plt.title('Learning Curves')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid(True)
plt.show()