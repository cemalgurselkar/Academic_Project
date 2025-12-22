import cv2
import os


class FrameExtractor:

    def __init__(self, video_path: str, output_folder: str):

        self.video_path = video_path
        self.output_folder = output_folder
        self.cap = None
        self.fps = 0.0

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def open_video(self) -> bool:
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Hata: {self.video_path} açılamadı.")
            return False
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Video FPS: {self.fps}")
        return True

    def extract_frames(self, frame_per_second: int = 1):

        if self.cap is None:
            raise RuntimeError("Video açılmamış. 'open_video()' çağrılmalı.")

        frame_interval = int(self.fps / frame_per_second)
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                save_path = os.path.join(
                    self.output_folder, f'frame_{saved_count}.png'
                )
                cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                saved_count += 1

            frame_count += 1

        print(f"Toplam {saved_count} kare kaydedildi.")

    def release(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file = '/home/cemal/My_Academic_Projects/Fish_Projects/Scenario-3/3/gopro12/GX011516.MP4'
    output_dir = 'Images_Data'

    extractor = FrameExtractor(video_file, output_dir)

    if extractor.open_video():
        fps_req = int(input("Saniyede kaç kare kaydedilsin? "))
        extractor.extract_frames(fps_req)
        extractor.release()