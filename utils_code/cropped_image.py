import cv2
import os

image_folder = "/home/cemal/My_Academic_Projects/Fish_Projects/utils_code/Images_Data(For_3)"
save_folder = "Cropped_Images(For_3)"

os.makedirs(save_folder, exist_ok=True)


images_file = [f for f in os.listdir(image_folder) if f.endswith(".png")]
print(len(images_file))
first_img = os.path.join(image_folder, images_file[0])
img = cv2.imread(first_img)
if img is None:
    raise RuntimeError("Resim yüklenemiyor.")

try:
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    print("Görsel oluşturuldu")    
    bbox = cv2.selectROI('img',img, showCrosshair=False, fromCenter=False)
    cv2.destroyAllWindows()

    x,y,w,h, = bbox
    print(f"Seçilen ROI: x={x}, y={y}, h={h}, w={w}")

    for img_file in images_file:
        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Yüklenemedi: {img_file}")
            continue

        cropped_img = img[int(y):int(y+h), int(x):int(x+w)]

        save_path = os.path.join(save_folder, img_file)
        cv2.imwrite(save_path, cropped_img)
        print(f"Kaydedildi: {save_path}")

    print("Tüm resimler kaydedildi!!")
except Exception as e:
    print(f"Error: {e}")

finally:
    cv2.destroyAllWindows()