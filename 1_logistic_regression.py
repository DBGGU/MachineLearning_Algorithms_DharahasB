# 1_logistic_regression.py
# Logistic Regression on Iris dataset

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print accuracy
print("Logistic Regression Accuracy on Iris dataset:", accuracy_score(y_test, predictions))
