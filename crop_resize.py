import os 
from PIL import Image

# -- Calculation -- #
# ImageJ value 
x, y, w, h = 1848, 972, 330, 780 # From ImageJ 
x_pixel, y_pixel = 4056, 3040 # 4k Pixels Resolution 

x1 = (x / x_pixel) 
x2 = ((x + w) / x_pixel) 
y1 = (y / y_pixel) 
y2 = ((y + h) / y_pixel)  

print(f"Crop coordinates: ({x1:.2f}, {x2:.2f}) to ({y1:.2f}, {y2:.2f})")


# -- Cropping and Resizing -- #
# CONFIGURATIONS # 
INPUT_DIR = "Data_Masked_Noisy" # Directory containing original images
OUTPUT_DIR = "Data_Resized" # Directory to save cropped and resized images
TARGET_SIZE = (224, 224) # Desired output size (width, height)

SPLITS = ["Train", "Test", "Validation"] # Dataset splits
CLASS_NAMES = ["Breast", "Prostate", "Skin", "Control"] # Class names

for split in SPLITS:
    for class_name in CLASS_NAMES:
        input_path = os.path.join(INPUT_DIR, split, class_name)
        output_path = os.path.join(OUTPUT_DIR, split, class_name)

        os.makedirs(output_path, exist_ok=True) # Create output directory if it doesn't exist
        
        for filename in os.listdir(input_path):
            if filename.endswith((".jpg", ".jpeg", ".png")): # Process only image files 
                in_img_path = os.path.join(input_path, filename)
                out_img_path = os.path.join(output_path, filename)

                img = Image.open(in_img_path).convert("RGB") # Ensure image is in RGB format

                # Crop the image using calculated coordinates
                width, height = img.size
                left = int(x1 * width)
                right = int(x2 * width)
                top = int(y1 * height)
                bottom = int(y2 * height)

                cropped = img.crop((left, top, right, bottom)) # Crop the image

                img_resize = cropped.resize(TARGET_SIZE)
                img_resize.save(out_img_path) # Save the processed image
                # print(f"Processed and saved: {out_img_path}")