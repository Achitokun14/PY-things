# Challenge
import numpy as np
from PIL import Image, ImageFilter

image = Image.open("in-case-of-fire.jpg")
image_array = np.array(image)
print(image_array.shape)
print(image_array.dtype)

r_channel = image_array[:, :, 0] * 1.2
g_channel = image_array[:, :, 1]
b_channel = image_array[:, :, 2] * 0.9

filtered_image = np.stack([r_channel, g_channel, b_channel], axis=-1).astype(np.uint8)

blurred_image = Image.fromarray(filtered_image)
blurred_image = blurred_image.filter(ImageFilter.GaussianBlur(radius=1))

blurred_image.show()