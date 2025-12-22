from PIL import Image
import os

input_folder = "Second_data"
output_folder = "Second_data"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(output_folder):
    if filename.endswith(".jpg"):
        jpg_path = os.path.join(input_folder, filename)
        png_path = os.path.join(output_folder, filename.replace(".jpg",".png"))


        image = Image.open(jpg_path)
        image.save(png_path, "PNG")