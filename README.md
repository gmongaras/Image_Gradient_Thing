# What is this
This repo is basically just a script that takes in an image, computes the gradient at some sampled points and puts a line orthogonal to that gradient according to the color of the image at that pixel.

There is also a video version where the lines are consistent through multiple frames. The lines rotate and change colors based on the movement in the video.


# Scripts

## image_grad.py

Takes an image as input and saves an image with the gradient.
- scale - Length of each line
- line_width - width of each line
- step_size - number of pixels between each line (lower resolution)



## image_grad_vide.py

Takes a video as input and converts each image using the script above. However, there is consistency between the lines such that the lines rotate and the color slowly faded.
- scale - Length of each line
- line_width - width of each line
- step_size - number of pixels between each line (lower resolution)
- rotation_factor - How quick the lines rotate to align with the gradient of the current frame
- fade_factor - How quick the lines change color to the current frame color



## to_4k.py

Takes a video as input and saves a 4K version. Just needed so I could upload to YouTube.