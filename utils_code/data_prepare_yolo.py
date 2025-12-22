import json
import os
from shutil import copyfile

json_file = 'Fish_Datasets/labelme'
image_file = 'Fish_Datasets/labelme'  
output_dir = 'output'
output_image_dir = 'images/' 

def normalize_coordinates(points, img_width, img_height):
    x_min = min(points[0][0], points[1][0])
    x_max = max(points[0][0], points[1][0])
    y_min = min(points[0][1], points[1][1])
    y_max = max(points[0][1], points[1][1])

    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    return x_center, y_center, width, height

def prepare_data_yolo(json_file, image_file, output_dir, output_image_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    img_width, img_height = 720, 1280 

    group_id_counter = 0

    image_name = os.path.splitext(os.path.basename(image_file))[0]
    output_txt = os.path.join(output_dir, image_name + '.txt')

    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    copyfile(image_file, os.path.join(output_image_dir, os.path.basename(image_file)))

    with open(output_txt, 'w') as f:
        for item in data['shapes']:

            if item['group_id'] is None:
                item['group_id'] = group_id_counter
                group_id_counter += 1

            label = item['label']
            points = item['points']

            label_dict = {"mullus_barbatus": 0, "Solea_solea": 1, "trachurus_trachurus": 2,"S_aurata":3}
            label_id = label_dict.get(label, -1)
            
            if label_id == -1:
                print(f"Uyarı: '{label}' etiketi tanimli değil.")
            
            x_center, y_center, width, height = normalize_coordinates(points, img_width, img_height)

            f.write(f"{label_id} {x_center} {y_center} {width} {height}\n")

    print(f"YOLO formatinda etiket dosyasi '{output_txt}' olarak kaydedildi.")

for filename in os.listdir(json_file):
    if filename.endswith(".json"):
        json_file = os.path.join(json_file, filename)
        image_file = os.path.join(image_file, os.path.splitext(filename)[0] + '.jpg')
        prepare_data_yolo(json_file, image_file, output_dir, output_image_dir)
        json_file = 'Fish_Datasets/labelme'
        image_file = 'Fish_Datasets/labelme'