import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Function to overlay orthogonal gradient lines with adjustable sparsity
def overlay_sparse_orthogonal_gradient(image_rgb, grad_x, grad_y, scale=2, line_width=2, step_size=10):
    h, w, _ = image_rgb.shape
    blank_background = np.zeros_like(image_rgb)  # Create a blank image

    for y in range(0, h, step_size):  # Adjust step size for sparsity
        for x in range(0, w, step_size):
            # Get the gradient direction and magnitude
            dx = grad_x[y, x, 0]
            dy = grad_y[y, x, 0]
            magnitude = np.sqrt(dx**2 + dy**2)
            
            # Normalize the gradient to determine the angle
            angle = np.arctan2(dy, dx)

            # Make the angle orthogonal
            orthogonal_angle = angle - np.pi / 2

            # Determine the start and end points of the line
            x1 = int(x - scale * np.cos(orthogonal_angle))
            y1 = int(y - scale * np.sin(orthogonal_angle))
            x2 = int(x + scale * np.cos(orthogonal_angle))
            y2 = int(y + scale * np.sin(orthogonal_angle))

            # Get the color of the pixel
            color = tuple(int(c) for c in image_rgb[y, x])

            # Draw the line with specified width on the blank background
            cv2.line(blank_background, (x1, y1), (x2, y2), color, line_width)

    return blank_background

# Load the image
image_path = 'img.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Convert to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Compute gradients
grad_x = cv2.Sobel(image_rgb, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(image_rgb, cv2.CV_64F, 0, 1, ksize=3)

# Overlay sparse orthogonal gradient lines
sparse_orthogonal_gradient_lines = overlay_sparse_orthogonal_gradient(
    image_rgb, grad_x, grad_y, scale=10, line_width=4, step_size=5
)

# Display the sparse orthogonal gradient lines
Image.fromarray(sparse_orthogonal_gradient_lines).save(f'out_{image_path}')