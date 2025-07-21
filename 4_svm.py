# 4_svm.py
# Support Vector Machine on Breast Cancer dataset

from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the SVM model
model = SVC()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print accuracy
print("SVM Accuracy on Breast Cancer dataset:", accuracy_score(y_test, predictions))
