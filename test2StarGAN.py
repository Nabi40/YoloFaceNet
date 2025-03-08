import os
from PIL import Image

input_dir = "data/img_align_celeba/img_align_celeba"
output_dir = "data/celeba_resized"
os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(input_dir):
    img = Image.open(os.path.join(input_dir, img_name))
    img = img.resize((128, 128))
    img.save(os.path.join(output_dir, img_name))

print("Resizing complete!")
