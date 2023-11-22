# Given the previous results, a Support Vector Machine (SVM) model could be a strong candidate. SVMs are known for their effectiveness in high-dimensional spaces and for complex classification problems. They work well when there is a clear margin of separation in the data.

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# The data preparation steps remain the same
data = np.load('data.npz')
X = data['features']
y = data['labels']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Support Vector Machine model
svm_classifier = SVC(random_state=0)

# Training the model
svm_classifier.fit(X_train_scaled, y_train)

# Evaluating the model
svm_accuracy = svm_classifier.score(X_test_scaled, y_test)

print(svm_accuracy)
