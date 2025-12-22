import cv2
import os
import argparse

class ImageCropper:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.output_folder = "Cropped_Images"
    
    def _prepare_files(self):
        images_file = [f for f in os.listdir(self.input_folder) if f.endswith(".png")]
        print(f"There is {len(images_file)} images in the folder")
        return images_file
    
    def _cropping(self):
        images_files = self._prepare_files()
        os.makedirs(self.output_folder, exist_ok=True)

        first_img = os.path.join(self.input_folder, images_files[0])
        img = cv2.imread(first_img)
        if img is None:
            raise RuntimeError("Can't load Image")
        
        try:
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            
            bbox = cv2.selectROI('img', img, showCrosshair=False, fromCenter=False)
            cv2.destroyAllWindows()
            x,y,w,h = bbox

            for img_file in images_files:
                img_path = os.path.join(self.input_folder, img_file)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Can't load: {img_file}")
                    continue
                cropped_img = img[int(y):int(y+h), int(x):int(x+w)]

                save_path = os.path.join(self.output_folder, img_file)
                cv2.imwrite(save_path, cropped_img)
                print(f"Kaydedildi: {save_path}")
            
            print("Tüm resimler kaydedildi!!")


        except Exception as e:
            print(f"Error message: {e}")
        
        finally:
            cv2.destroyAllWindows()
    
    def run(self):
        try:
            self._cropping()
        except Exception as e:
            print(f"Error message: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch crop images using the same ROI")
    parser.add_argument("input_folder", "help=Path to folder containing images")
    args = parser.parse_args()

    cropper = ImageCropper(args.input_folder)
    cropper.run()