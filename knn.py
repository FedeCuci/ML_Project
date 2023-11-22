import numpy as np
import os
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

data = np.load('data.npz')
X = data['features']
y = data['labels']
neighbors_to_try = [i for i in range(1,75,2)]

# Create empty lists to save accuracies
training_accuracy = []
test_accuracy = []

X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, random_state=0)

for n_neighbors in neighbors_to_try:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    training_accuracy.append(knn.score(X_train, y_train))
    # Save test accuracy in our list
    test_accuracy.append(knn.score(X_test, y_test))


# plt.plot(neighbors_to_try, training_accuracy, label="training accuracy")
# plt.plot(neighbors_to_try, test_accuracy, label="test accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("n_neighbors")
# plt.legend()
# plt.show()

y_pred = knn.predict(X_test)

print(np.mean(y_pred == y_test))
print(knn.score(X_test, y_test))

print(test_accuracy)
