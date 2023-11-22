# Re-attempting to improve the Random Forest classifier accuracy with hyperparameter tuning

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Loading and preparing the data
data = np.load('data.npz')
X = data['features']
y = data['labels']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating and training the Random Forest model with default settings
rf_classifier = RandomForestClassifier(random_state=0)
rf_classifier.fit(X_train_scaled, y_train)

# Evaluating the model with default settings
rf_accuracy_default = rf_classifier.score(X_test_scaled, y_test)

# Tuning the model by adjusting hyperparameters
rf_classifier_tuned = RandomForestClassifier(
    n_estimators=200,  # Increased number of trees
    max_depth=20,      # Setting max depth for each tree
    random_state=0
)

# Training the tuned model
rf_classifier_tuned.fit(X_train_scaled, y_train)

# Evaluating the tuned model
rf_accuracy_tuned = rf_classifier_tuned.score(X_test_scaled, y_test)

print(rf_accuracy_default, rf_accuracy_tuned)