import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np

# SPECT PET CT
mri_folder = "Your MRI Foder"
# D://RESEARCH/IMAGE FUSION/FUSION DATASET/PET-MRI/test/PET/
spect_folder = "Your SPECT Foder"
fused_folder = "Your FUSED IMAGE Foder"
output_folder = "SPECIFY THE OUTPUT FOLDER"
img_size = 256

# Define zoom regions (x_start, y_start, width, height)
zoom_regions = [
    (150, 100, 30, 30),  # First zoom region
    (70, 180, 30, 30),  # Second zoom region
]

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Display images from MRI, SPECT, and FUSED folders side by side
mri_images = sorted(os.listdir(mri_folder))
spect_images = sorted(os.listdir(spect_folder))
fused_images = sorted(os.listdir(fused_folder))

# Ensure all folders have the same number of images
num_images = min(len(mri_images), len(spect_images), len(fused_images))

for i in range(num_images):
    mri_image = Image.open(os.path.join(mri_folder, mri_images[i])).resize((img_size, img_size))
    spect_image = Image.open(os.path.join(spect_folder, spect_images[i])).resize((img_size, img_size))
    fused_image = Image.open(os.path.join(fused_folder, fused_images[i])).resize((img_size, img_size))

    # Convert all images to numpy arrays (ensure they are RGB)
    mri_image_np = np.array(mri_image)
    spect_image_np = np.array(spect_image)
    fused_image_np = np.array(fused_image)

    if len(mri_image_np.shape) == 2:  # Convert grayscale to RGB
        mri_image_np = np.stack([mri_image_np] * 3, axis=-1)
    if len(spect_image_np.shape) == 2:
        spect_image_np = np.stack([spect_image_np] * 3, axis=-1)
    if len(fused_image_np.shape) == 2:
        fused_image_np = np.stack([fused_image_np] * 3, axis=-1)

    # Create a subplot for each set of images (one row for MRI, SPECT, FUSED)
    fig, axs = plt.subplots(2, 3, figsize=(8, 6))

    # Display MRI image
    axs[0, 0].imshow(mri_image_np)
    axs[0, 0].axis('off')
    lw = 2
    for (x, y, w, h) in zoom_regions:
        rect = Rectangle((x, y), w, h, edgecolor='red', facecolor='none', lw=lw)
        axs[0, 0].add_patch(rect)

    # Zoomed regions for MRI with border
    mri_zoomed_combined = []
    for (x, y, w, h) in zoom_regions:
        mri_zoomed_region = mri_image_np[y:y + h, x:x + w, :]
        border_thickness = 1
        mri_zoomed_with_border = np.pad(
            mri_zoomed_region,
            pad_width=((border_thickness, border_thickness), (border_thickness, border_thickness), (0, 0)),
            mode="constant",
            constant_values=0
        )
        mri_zoomed_with_border[:border_thickness, :, 0] = 255
        mri_zoomed_with_border[-border_thickness:, :, 0] = 255
        mri_zoomed_with_border[:, :border_thickness, 0] = 255
        mri_zoomed_with_border[:, -border_thickness:, 0] = 255
        mri_zoomed_combined.append(mri_zoomed_with_border)
    axs[1, 0].imshow(np.concatenate(mri_zoomed_combined, axis=1))
    axs[1, 0].axis('off')

    # Display SPECT image
    axs[0, 1].imshow(spect_image_np)
    axs[0, 1].axis('off')
    for (x, y, w, h) in zoom_regions:
        rect = Rectangle((x, y), w, h, edgecolor='red', facecolor='none', lw=lw)
        axs[0, 1].add_patch(rect)

    # Zoomed regions for SPECT with border
    spect_zoomed_combined = []
    for (x, y, w, h) in zoom_regions:
        spect_zoomed_region = spect_image_np[y:y + h, x:x + w, :]
        border_thickness = 1
        spect_zoomed_with_border = np.pad(
            spect_zoomed_region,
            pad_width=((border_thickness, border_thickness), (border_thickness, border_thickness), (0, 0)),
            mode="constant",
            constant_values=0)
        spect_zoomed_with_border[:border_thickness, :, 0] = 255
        spect_zoomed_with_border[-border_thickness:, :, 0] = 255
        spect_zoomed_with_border[:, :border_thickness, 0] = 255
        spect_zoomed_with_border[:, -border_thickness:, 0] = 255
        spect_zoomed_combined.append(spect_zoomed_with_border)
    axs[1, 1].imshow(np.concatenate(spect_zoomed_combined, axis=1))
    axs[1, 1].axis('off')

    # Display FUSED image
    axs[0, 2].imshow(fused_image_np)
    axs[0, 2].axis('off')
    for (x, y, w, h) in zoom_regions:
        rect = Rectangle((x, y), w, h, edgecolor='red', facecolor='none', lw=lw)
        axs[0, 2].add_patch(rect)

    # Zoomed regions for FUSED with border
    fused_zoomed_combined = []
    for (x, y, w, h) in zoom_regions:
        fused_zoomed_region = fused_image_np[y:y + h, x:x + w, :]
        border_thickness = 1
        fused_zoomed_with_border = np.pad(
            fused_zoomed_region,
            pad_width=((border_thickness, border_thickness), (border_thickness, border_thickness), (0, 0)),
            mode="constant",
            constant_values=0
        )
        fused_zoomed_with_border[:border_thickness, :, 0] = 255
        fused_zoomed_with_border[-border_thickness:, :, 0] = 255
        fused_zoomed_with_border[:, :border_thickness, 0] = 255
        fused_zoomed_with_border[:, -border_thickness:, 0] = 255
        fused_zoomed_combined.append(fused_zoomed_with_border)
    axs[1, 2].imshow(np.concatenate(fused_zoomed_combined, axis=1))
    axs[1, 2].axis('off')

    # Adjust layout for saving
    plt.subplots_adjust(hspace=-0.75, wspace=0)
    plt.tight_layout(pad=0.5)

    # Save the figure to the output folder with minimized white space
    output_path = os.path.join(output_folder, f"output_{i+1}.jpeg")
    plt.savefig(output_path, format='jpeg', dpi=500, bbox_inches='tight',pad_inches=0.1)
    plt.close(fig)  # Close the figure to save memory
