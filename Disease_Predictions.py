# Step 1: Load necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 2: Load the dataset
heart_data = pd.read_csv("heart.csv")

# Step 3: Drop faulty data entries
faulty_entries = [93, 159, 164, 165, 252, 49, 282]
heart_data = heart_data.drop(faulty_entries)

# Step 4: Split the dataset into features and target
X = heart_data.drop('target', axis=1)
y = heart_data['target']

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Decision Trees and Random Forest models
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Step 7: Evaluate model performance
dt_pred = dt_classifier.predict(X_test)
rf_pred = rf_classifier.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Decision Trees Accuracy:", dt_accuracy)
print("Random Forest Accuracy:", rf_accuracy)

# Step 8: Assess the importance of different features
feature_importances = pd.DataFrame(rf_classifier.feature_importances_,
                                   index=X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print("Feature Importances:")
print(feature_importances)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importances.index, feature_importances['importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - Random Forest')
plt.show()

# Step 9: Compare the results of Decision Trees and Random Forests
print("\nClassification Report - Decision Trees:")
print(classification_report(y_test, dt_pred))

print("\nClassification Report - Random Forest:")
print(classification_report(y_test, rf_pred))

# Plot Confusion Matrix for Decision Trees
disp = ConfusionMatrixDisplay.from_predictions(y_test, dt_pred)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Decision Trees')
plt.show()

# Plot Confusion Matrix for Random Forest
disp = ConfusionMatrixDisplay.from_predictions(y_test, rf_pred)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Random Forest')
plt.show()

# Plot Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=['No Disease', 'Disease'])
plt.title('Decision Tree Visualization')
plt.show()
