import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
n_features = 5

# Create synthetic features (e.g., age, tumor size, etc.)
X = np.random.rand(n_samples, n_features)

# Create synthetic labels (0 for no cancer, 1 for cancer)
y = (X[:, 0] + X[:, 1] + X[:, 2] > 1.5).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print results
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Example of using the model for a new patient
new_patient = np.random.rand(1, n_features)
prediction = model.predict(new_patient)
print("\nNew Patient Prediction:")
print("Cancer" if prediction[0] == 1 else "No Cancer")