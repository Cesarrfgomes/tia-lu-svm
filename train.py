import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

print("Loading processed data...")
df = pd.read_csv('dataset/processed_data.csv')

X = df.drop('target', axis=1)
y = df['target']

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training LinearSVC...")
model = LinearSVC(class_weight='balanced', random_state=42, max_iter=10000, dual=False)
model.fit(X_train, y_train)

print("Evaluating...")
y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['DELIVERED', 'CANCELLED']))

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

joblib.dump(model, 'svm_model.pkl')
