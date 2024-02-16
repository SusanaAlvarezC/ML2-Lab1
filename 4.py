import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Get the path where the script is located
current_directory = os.path.dirname(os.path.abspath(__file__))
unsupervised_path = os.path.join(current_directory, 'unsupervised')
sys.path.append(unsupervised_path)

from unsupervised.svd import SVD

image_path = os.path.join(current_directory, 'Resources/Fotos ML2/susana_alvarez.jpg')

# Load the image into the original_image variable
original_image = np.array(Image.open(image_path).convert('L'))

# Ask the user for the number of components
n_components = int(input("Enter the number of components for SVD decomposition: "))

# Create an instance of the SVD class with the provided number of components
from unsupervised.svd import SVD
svd = SVD(n_components)

# Transform the original matrix using the SVD class
reconstructed_matrix = svd.fit_transform(original_image)

# Create an image from the reconstructed matrix
reconstructed_image = Image.fromarray(reconstructed_matrix.astype('uint8'), 'L')

# Show the image and the Euclidean distance
plt.imshow(reconstructed_image, cmap='gray')
plt.title(f'SV = {n_components}')
plt.show()
