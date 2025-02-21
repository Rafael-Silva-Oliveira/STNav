# %%
import numpy as np
from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
from PIL import Image
import squidpy as sq

# Load the image
img = Image.open("/mnt/work/RO_src/data/raw/_SFH173_20__HE.vsi - 40x_BF_01.tif")

# Example coordinates for the region of interest (ROI)
# These would typically be obtained from cell detection or prior knowledge.
roi_top_left_x = 3400  # Example x coordinate for ROI
roi_top_left_y = 2700  # Example y coordinate for ROI

# Define the desired crop size (zoom level)
# Make sure the crop size is not larger than the ROI
zoom_crop_size = (600, 600)  # Example smaller crop size for zoom

# Calculate the bottom-right corner based on the new crop size
roi_bottom_right_x = roi_top_left_x + zoom_crop_size[0]
roi_bottom_right_y = roi_top_left_y + zoom_crop_size[1]

# Ensure the crop does not exceed the image dimensions
img_width, img_height = img.size
roi_bottom_right_x = min(roi_bottom_right_x, img_width)
roi_bottom_right_y = min(roi_bottom_right_y, img_height)

# Adjust the top-left corner if the cropping box exceeds the image dimensions
roi_top_left_x = max(roi_top_left_x, 0)
roi_top_left_y = max(roi_top_left_y, 0)

# Define the cropping box with adjusted coordinates
crop_box = (roi_top_left_x, roi_top_left_y, roi_bottom_right_x, roi_bottom_right_y)

# Crop the image using the adjusted box
cropped_img = img.crop(crop_box)

cropped_img = np.array(cropped_img)  # Convert to numpy array

# Plot the results
plt.subplot(1, 2, 1)
plt.imshow(cropped_img, cmap="gray")
plt.axis("off")
plt.title("Cropped Input Image")
# %%
# Normalize the image
cropped_img = normalize(cropped_img)
# Load the pretrained StarDist model for H&E
model = StarDist2D.from_pretrained("2D_versatile_he")

# %%
# Predict instances with the correct axes specified
labels, _ = model.predict_instances(cropped_img, axes="YXC")


# %%
plt.subplot(1, 2, 2)
plt.imshow(render_label(labels))
plt.axis("off")
plt.title("Prediction + Input Overlay")
plt.show()

# %%
