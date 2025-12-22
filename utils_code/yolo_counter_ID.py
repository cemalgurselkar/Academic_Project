import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r"C:\Users\Cemal\academic_projects\runs\detect\train\weights\best.pt"
VIDEO_PATH = r"C:\Users\Cemal\academic_projects\Fish_Projects\cemal_örnek_veri\Scenario-1\5\GoPro12\GX011530.MP4"

def select_roi(window_name: str, frame: np.ndarray) -> tuple[int,int,int,int]:
    """Fare ile ROI seçtir ve (x,y,w,h) döndür."""
    roi = cv2.selectROI(window_name, frame, showCrosshair=False, fromCenter=False)
    cv2.destroyWindow(window_name)
    x, y, w, h = map(int, roi)
    if w == 0 or h == 0:
        raise RuntimeError("Geçerli bir ROI seçilmedi.")
    return x, y, w, h

def main():
    # Modeli yükle
    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Video açılamadı:", VIDEO_PATH)
        return

    roi = None
    counted_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF

        # 1) 'r' tuşuna basınca ROI seç
        if key == ord('r'):
            try:
                roi = select_roi("ROI Seç", frame)
                counted_ids.clear()
                print("Yeni ROI:", roi)
            except RuntimeError as e:
                print(e)
                roi = None

        # 2) ROI varsa takip et ve say
        if roi is not None:
            x, y, w, h = roi
            # Tüm frame'de nesne takibi
            results = model.track(frame, persist=True)[0]
            boxes = results.boxes.xyxy.cpu().numpy()        # type: ignore
            ids   = (results.boxes.id.cpu().numpy()         # type: ignore
                     if results.boxes.id is not None else []) # type: ignore

            # ROI sınırlarını çiz
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

            # Her kutu için merkez ve sayım
            for box, obj_id in zip(boxes, ids):
                obj_id = int(obj_id)
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2)//2, (y1 + y2)//2

                if x <= cx <= x + w and y <= cy <= y + h:
                    if obj_id not in counted_ids:
                        counted_ids.add(obj_id)
                    # Kutu ve ID çiz
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{obj_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Toplam sayıyı yaz
            cv2.putText(frame, f"Total Fish: {len(counted_ids)}",
                        (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # 3) Göster ve çıkış tuşu
        cv2.imshow("Fish Counter", frame)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
