from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


# Get and print the path where the image is located
current_directory = os.path.dirname(os.path.abspath(__file__))
print(current_directory)

# Concatenate the path and file name of the image
image_path = os.path.join(current_directory, 'Resources', 'susana_alvarez_origin.jpg')

print(image_path)

# Load the image into the variable original_image
original_image = Image.open(image_path)

# Convert the image to grayscale and resize it to 256x256 pixels
bw_image = original_image.convert('L')
resized_image = bw_image.resize((256, 256))

# Specify the relative path to the 'Resources\Fotos ML2' folder
relative_path = 'Resources\Fotos ML2'

# Combine the current directory and the relative path to get the full path
images_directory = os.path.join(current_directory, relative_path)

# Save the image locally
local_image_path = os.path.join(images_directory, 'susana_alvarez.jpg')
resized_image.save(local_image_path)

# Load all images from the directory
image_files = [f for f in os.listdir(images_directory) if f.endswith('.jpg')]
print(image_files)

# Ensure at least one image is found
if not image_files:
    print("No image files were found in the specified directory")
else:
     # Load the first image to get dimensions
    first_image_path = os.path.join(images_directory, image_files[0])
    first_image = Image.open(first_image_path)
    width, height = first_image.size

     # Initialize an array to store pixel values
    pixel_sum = np.zeros((height, width))

    # Loop through each image and accumulate pixel values
    for image_file in image_files:
        image_path = os.path.join(images_directory, image_file)
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = image.resize((width, height))  # Resize to consistent dimensions
        pixels = np.array(image)
        pixel_sum += pixels

    # Calculate the average pixel value
    average_pixels = pixel_sum / len(image_files)    
    print(average_pixels)
   
    average_face = Image.fromarray(average_pixels.astype('uint8'))    
   

# Calculate pixel-wise difference between resized image and average face
pixel_difference = resized_image - average_pixels

# Compute the Euclidean norm
euclidean_distance = np.linalg.norm(pixel_difference)

# Print the Euclidean distance
print(f"Distancia Euclidiana: {euclidean_distance}")

# Initialize an array to store Euclidean distances for each image
euclidean_distances = []

# Loop through each image and calculate the Euclidean distance
for image_file in image_files:
    image_path = os.path.join(images_directory, image_file)
    image = Image.open(image_path).convert('L')  # Convertir a escala de grises
    image = image.resize((width, height))  # Redimensionar a dimensiones consistentes
    pixels = np.array(image)

    # Calculate the pixel-wise difference between the resized image and the current image
    pixel_difference = pixels - average_pixels

    # Calculate the Euclidean norm
    euclidean_distance = np.linalg.norm(pixel_difference)
    euclidean_distances.append(euclidean_distance)

    print(f"Euclidean Distance for  {image_file}: {euclidean_distance}")


# Show the Euclidean distances with labels and colors
bar_labels = ['' for _ in image_files]  # Initialize all labels as empty
susana_index = image_files.index('susana_alvarez.jpg')
bar_labels[susana_index] = 'susana_alvarez.jpg'  # Add the label only for 'susana_alvarez.jpg'

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Show the resized image
axes[0].imshow(resized_image, cmap='gray')
axes[0].set_title('Resized Image')

# Mostrar la cara promedio
axes[1].imshow(average_face, cmap='gray')
axes[1].set_title('Average Face of the Group')

# Show the pixel-wise difference
pixel_difference = resized_image - average_pixels
axes[2].imshow(pixel_difference, cmap='bwr', vmin=-95, vmax=95)  # Ajustar vmin y vmax para una mejor visualización
axes[2].set_title('Pixel-wise Difference')


# Show the Euclidean distances with labels and colors
bars = axes[3].bar(image_files, euclidean_distances, tick_label=bar_labels, color='lightgray')
bars[susana_index].set_color('blue')  # Cambiar el color de la barra de 'susana_alvarez.jpg'
axes[3].set_title('Euclidean Distances')

# Show the plot
plt.show() 