import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Create a small custom dataset
# Features: smile_width (0-1), eyebrow_angle (0-1), eye_openness (0-1)
# Emotions: Happy, Sad, Angry
data = {
    'smile_width': [0.8, 0.7, 0.9, 0.2, 0.1, 0.3, 0.4, 0.5, 0.6, 0.2, 0.3, 0.4],
    'eyebrow_angle': [0.2, 0.3, 0.1, 0.7, 0.8, 0.6, 0.5, 0.4, 0.3, 0.8, 0.7, 0.6],
    'eye_openness': [0.7, 0.6, 0.8, 0.4, 0.3, 0.5, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    'emotion': ['Happy', 'Happy', 'Happy', 'Sad', 'Sad', 'Sad', 'Angry', 'Angry', 'Angry', 'Sad', 'Happy', 'Angry']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Prepare features (X) and labels (y)
X = df[['smile_width', 'eyebrow_angle', 'eye_openness']]
y = df['emotion']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# Example prediction with new data
new_data = np.array([[0.85, 0.2, 0.7]])  # Example: wide smile, low eyebrow angle, open eyes
predicted_emotion = model.predict(new_data)
print(f"\nPredicted emotion for new data (smile_width=0.85, eyebrow_angle=0.2, eye_openness=0.7): {predicted_emotion[0]}")