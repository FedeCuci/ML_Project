# For the new model, I'll use a Random Forest Classifier. This model is suitable because:
# - It handles both numerical and categorical data well.
# - It's less sensitive to outliers than KNN.
# - It can capture complex patterns in the data.
# - It typically performs well in multi-class classification tasks.
# - It has in-built feature selection, which can be beneficial.

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay





# The data loading, imputation, and scaling steps remain the same as for KNN
data = np.load('data.npz')
X = data['features']
y = data['labels']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Random Forest model
rf_classifier = RandomForestClassifier(
    random_state=0,
    n_estimators=500,
    max_depth=20,
    max_features='log2'
    # min_samples_leaf=2,
    # min_samples_split=2
    )
rf_classifier.fit(X_train_scaled, y_train)

# Evaluate the model
rf_accuracy = rf_classifier.score(X_test_scaled, y_test)

y_pred = rf_classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


# print(classification_report(y_test, y_pred))
print(rf_accuracy)
