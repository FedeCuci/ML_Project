import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score


# Load the data
data = np.load('data.npz')
X = data['features']
y = data['labels']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, random_state=0)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test the KNN model with the scaled data
neighbors_to_try = [i for i in range(1, 75, 2)]
training_accuracy = []
test_accuracy = []

for n_neighbors in neighbors_to_try:
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights='distance',
        algorithm='ball_tree'
        )
    knn.fit(X_train_scaled, y_train)

    # Save training and test accuracy
    training_accuracy.append(knn.score(X_train_scaled, y_train))
    test_accuracy.append(knn.score(X_test_scaled, y_test))

# Find the number of neighbors with the highest test accuracy
optimal_neighbors = neighbors_to_try[test_accuracy.index(max(test_accuracy))]
max_accuracy = max(test_accuracy)

print(optimal_neighbors, max_accuracy)