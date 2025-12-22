import os
import shutil
import random

source_dir = 'ProjectForAcademic/output'
images_dir = 'images/'
labels_dir = 'labels/'

os.makedirs(os.path.join(images_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(images_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(labels_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(labels_dir, 'val'), exist_ok=True)

files = os.listdir(source_dir)

image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
txt_files = [f for f in files if f.endswith('.txt')]
random.shuffle(image_files)
split_idx = int(0.8 * len(image_files))

train_images = image_files[:split_idx]
val_images = image_files[split_idx:]

def move_files(image_list, target_image_dir, target_label_dir):
    for image in image_list:
        txt_file = image.replace('.jpg', '.txt').replace('.png', '.txt')

        shutil.copy(os.path.join(source_dir, image), os.path.join(target_image_dir, image))
        shutil.copy(os.path.join(source_dir, txt_file), os.path.join(target_label_dir, txt_file))

move_files(train_images, os.path.join(images_dir, 'train'), os.path.join(labels_dir, 'train'))
move_files(val_images, os.path.join(images_dir, 'val'), os.path.join(labels_dir, 'val'))

print("İmages and Labels folders are ready!!")
