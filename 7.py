import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA



# Download the MNIST dataset
mnist = fetch_openml('mnist_784')

# Filter images and labels only for digits 0 and 8
indices = np.where((mnist.target == '0') | (mnist.target == '8'))[0]
images = mnist.data.iloc[indices].to_numpy()
labels = mnist.target.iloc[indices].astype(int).to_numpy()

# Apply SVD with two components
n_components = 2
svd = TruncatedSVD(n_components)
images_svd = svd.fit_transform(images)

# Split the SVD dataset into training and test sets
X_train_svd, X_test_svd, y_train_svd, y_test_svd = train_test_split(images_svd, labels, test_size=0.2, random_state=42)

# Scale the data with dimensionality reduction using SVD
scaler_svd = StandardScaler()
X_train_scaled_svd = scaler_svd.fit_transform(X_train_svd)
X_test_scaled_svd = scaler_svd.transform(X_test_svd)

# Initialize and train the logistic regression model with SVD reduction
model_svd = LogisticRegression()
model_svd.fit(X_train_scaled_svd, y_train_svd)

# Make predictions on the test set with SVD reduction
predictions_svd = model_svd.predict(X_test_scaled_svd)

# Calculate and display evaluation metrics with SVD reduction
accuracy_svd = accuracy_score(y_test_svd, predictions_svd)
precision_svd = precision_score(y_test_svd, predictions_svd, pos_label=8)
recall_svd = recall_score(y_test_svd, predictions_svd, pos_label=8)
f1_svd = f1_score(y_test_svd, predictions_svd, pos_label=8)

print("\nWith dimensionality reduction (SVD):")
print(f'Accuracy: {accuracy_svd * 100:.2f}%')
print(f'Precision: {precision_svd * 100:.2f}%')
print(f'Recall: {recall_svd * 100:.2f}%')
print(f'F1 Score: {f1_svd * 100:.2f}%')

# Apply PCA with two components
n_components = 2
pca = PCA(n_components)
images_pca = pca.fit_transform(images)

# Split the PCA dataset into training and test sets
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(images_pca, labels, test_size=0.2, random_state=42)

# Scale the data with dimensionality reduction using PCA
scaler_pca = StandardScaler()
X_train_scaled_pca = scaler_pca.fit_transform(X_train_pca)
X_test_scaled_pca = scaler_pca.transform(X_test_pca)

# Initialize and train the logistic regression model with PCA reduction
model_pca = LogisticRegression()
model_pca.fit(X_train_scaled_pca, y_train_pca)

# Make predictions on the test set with PCA reduction
predictions_pca = model_pca.predict(X_test_scaled_pca)

# Calculate and display evaluation metrics with PCA reduction
accuracy_pca = accuracy_score(y_test_pca, predictions_pca)
precision_pca = precision_score(y_test_pca, predictions_pca, pos_label=8)
recall_pca = recall_score(y_test_pca, predictions_pca, pos_label=8)
f1_pca = f1_score(y_test_pca, predictions_pca, pos_label=8)

print("\nWith dimensionality reduction (PCA):")
print(f'Accuracy: {accuracy_pca * 100:.2f}%')
print(f'Precision: {precision_pca * 100:.2f}%')
print(f'Recall: {recall_pca * 100:.2f}%')
print(f'F1 Score: {f1_pca * 100:.2f}%')


# Plot the two new features generated by SVD
plt.figure(figsize=(10, 6))
plt.scatter(X_train_scaled_svd[:, 0], X_train_scaled_svd[:, 1], c=y_train_svd, cmap='viridis', marker='o', edgecolors='k', label='Train set')
plt.scatter(X_test_scaled_svd[:, 0], X_test_scaled_svd[:, 1], c=y_test_svd, cmap='viridis', marker='s', edgecolors='k', label='Test set')
plt.title('SVD: 2 New Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Plot the two new features generated by PCA
plt.figure(figsize=(10, 6))
plt.scatter(X_train_scaled_pca[:, 0], X_train_scaled_pca[:, 1], c=y_train_pca, cmap='viridis', marker='o', edgecolors='k', label='Train set')
plt.scatter(X_test_scaled_pca[:, 0], X_test_scaled_pca[:, 1], c=y_test_pca, cmap='viridis', marker='s', edgecolors='k', label='Test set')
plt.title('PCA: 2 New Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
