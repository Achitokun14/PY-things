# Exercise 4
import numpy as np
from PIL import Image
# import cv2

# a
image = Image.open("in-case-of-fire.jpg")
# image = cv2.imread("the-count-of-monte-cristo.png")
image_array = np.array(image)

# b
r_channel = image_array[:, :, 0]
g_channel = image_array[:, :, 1]
b_channel = image_array[:, :, 2]
# b_channel, g_channel, r_channel = cv2.split(image)

grayscale = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])
# grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

flipped_horizontal = np.fliplr(image_array)
flipped_vertical = np.flipud(image_array)
# flipped_horizontal = cv2.flip(image, 1)
# flipped_vertical = cv2.flip(image, 0)

brightness_factor = 1.2
adjusted_brightness = np.clip(image_array * brightness_factor, 0, 255).astype(np.uint8)

print("Image Properties:")
print(f"Dimensions: {image.size[0]}x{image.size[1]}")
print(f"Channels: {len(image.mode)}")
# print(f"Dimensions: {image.shape[:2]}")
# print(f"Channels: {image.shape[2]}")

print("\nOperations Complete:")
print(f"- RGB channels extracted")
print(f"- Grayscale conversion") 
print(f"- Image flipped") 
print(f"- Brightness adjusted")

'''
PS H:\DAISI\Python Programming - TP2> python '.\Exercise 4 - Image Processing.py'
Image Properties:
Dimensions: 800x1088
Channels: 3

Operations Complete:
- RGB channels extracted
- Grayscale conversion
- Image flipped
- Brightness adjusted
'''