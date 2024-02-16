from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Download the MNIST dataset
mnist = fetch_openml('mnist_784')

# Filter images and labels only for digits 0 and 8
indices = np.where((mnist.target == '0') | (mnist.target == '8'))[0]
images = mnist.data.iloc[indices].to_numpy()
labels = mnist.target.iloc[indices].astype(int).to_numpy()

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the logistic regression model with a maximum number of iterations
model = LogisticRegression()

# Train the model with scaled data
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
predictions = model.predict(X_test_scaled)

# Calculate and display evaluation metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, pos_label=8)
recall = recall_score(y_test, predictions, pos_label=8)
f1 = f1_score(y_test, predictions, pos_label=8)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')
