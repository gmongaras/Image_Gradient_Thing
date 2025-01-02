import cv2
import numpy as np

# Function to overlay sparse orthogonal gradient lines with adjustable sparsity, rotation, and fading colors
def overlay_sparse_orthogonal_gradient(image_rgb, grad_x, grad_y, angle_offsets, previous_colors, scale=2, line_width=2, step_size=10, rotation_factor=0.1, fade_factor=0.1, initialize=False):
    h, w, _ = image_rgb.shape

    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(0, h, step_size), np.arange(0, w, step_size), indexing="ij")

    # Get gradient values at sparse grid locations
    dx = grad_x[y_coords, x_coords, 0]
    dy = grad_y[y_coords, x_coords, 0]

    # Compute gradient angles
    angles = np.arctan2(dy, dx)
    orthogonal_angles = angles - np.pi / 2

    if initialize:
        # Initialize the angle offsets and previous colors
        angle_offsets[y_coords, x_coords] = orthogonal_angles
        previous_colors[y_coords, x_coords] = image_rgb[y_coords, x_coords]
    else:
        # Smoothly update the angle offsets and previous colors
        angle_offsets[y_coords, x_coords] += rotation_factor * (orthogonal_angles - angle_offsets[y_coords, x_coords])
        previous_colors[y_coords, x_coords] = (1 - fade_factor) * previous_colors[y_coords, x_coords] + fade_factor * image_rgb[y_coords, x_coords]

    # Compute line start and end points for the sparse grid
    x1 = (x_coords - scale * np.cos(angle_offsets[y_coords, x_coords])).astype(np.int32)
    y1 = (y_coords - scale * np.sin(angle_offsets[y_coords, x_coords])).astype(np.int32)
    x2 = (x_coords + scale * np.cos(angle_offsets[y_coords, x_coords])).astype(np.int32)
    y2 = (y_coords + scale * np.sin(angle_offsets[y_coords, x_coords])).astype(np.int32)

    # Clip the coordinates to stay within bounds
    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    x2 = np.clip(x2, 0, w - 1)
    y2 = np.clip(y2, 0, h - 1)

    # Prepare the blank canvas
    blank_background = np.zeros_like(image_rgb)

    # Vectorized line drawing
    for i in range(len(x_coords)):
        for j in range(len(x_coords[0])):
            color = tuple(int(c) for c in previous_colors[y_coords[i, j], x_coords[i, j]])
            cv2.line(blank_background, (x1[i, j], y1[i, j]), (x2[i, j], y2[i, j]), color, line_width)
            # blank_background[y:y+png.height, x:x+png.width] = (np.array(png)[:, :, :3] * (np.array(png)[:, :, 3:4] / 255) + blank_background[y:y+png.height, x:x+png.width] * (1 - np.array(png)[:, :, 3:4] / 255)).astype(np.uint8)




    return blank_background, angle_offsets, previous_colors




# Video processing
# video_path = 'input_video.mp4'  # Replace with your input video path
# output_path = 'output_video.mp4'  # Replace with your desired output video path
video_path = 'bad_apple_10.mp4'  # Replace with your input video path
output_path = 'bad_apple_10_out.mp4'  # Replace with your desired output video path

# Open the video
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create a video writer
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Initialize angle offsets and previous colors
angle_offsets = np.zeros((frame_height, frame_width), dtype=np.float32)
previous_colors = np.zeros((frame_height, frame_width, 3), dtype=np.float32)
first_frame = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Compute gradients
    grad_x = cv2.Sobel(frame_rgb, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(frame_rgb, cv2.CV_64F, 0, 1, ksize=3)

    # Overlay sparse orthogonal gradient lines with rotation and fading colors
    sparse_orthogonal_gradient_frame, angle_offsets, previous_colors = overlay_sparse_orthogonal_gradient(
        # frame_rgb, grad_x, grad_y, angle_offsets, previous_colors, scale=10, line_width=2, step_size=3, rotation_factor=0.25, fade_factor=0.3, initialize=first_frame
        frame_rgb, grad_x, grad_y, angle_offsets, previous_colors, scale=10, line_width=2, step_size=3, rotation_factor=0.1, fade_factor=0.1, initialize=first_frame
    )

    # After the first frame, disable initialization
    first_frame = False

    # Convert back to BGR for saving
    sparse_orthogonal_gradient_frame_bgr = cv2.cvtColor(sparse_orthogonal_gradient_frame, cv2.COLOR_RGB2BGR)

    # Write the frame to the output video
    out.write(sparse_orthogonal_gradient_frame_bgr)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved to {output_path}")
