import cv2
from ultralytics import YOLO
from ultralytics.solutions import ObjectCounter

MODEL_PATH = r"C:\Users\Cemal\academic_projects\runs\detect\train\weights\best.pt"
VIDEO_PATH = r"C:\Users\Cemal\academic_projects\Fish_Projects\cemal_örnek_veri\Scenario-1\5\GoPro12\GX011530.MP4"

line_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(line_points) < 2:
        line_points.append((x, y))
        print(f"Nokta {len(line_points)}: ({x}, {y})")

def select_line(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("İlk kare alınamadı.")

    window_name = "draw_a_line"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        temp = frame.copy()
        for pt in line_points:
            cv2.circle(temp, pt, 5, (0, 255, 255), -1)
        if len(line_points) == 2:
            cv2.line(temp, line_points[0], line_points[1], (255, 0, 255), 2)

        cv2.imshow(window_name, temp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cv2.destroyWindow(window_name)
    return line_points if len(line_points) == 2 else None

def main():
    global line_points
    region = select_line(VIDEO_PATH)
    if not region:
        print("Geçerli çizgi seçilmedi.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Video açılamadı.")
        return

    model = YOLO(MODEL_PATH)

    counter = ObjectCounter(
        model=r"C:\Users\Cemal\academic_projects\runs\detect\train\weights\best.pt",
        region=region,
        classes=[0],
        show=False
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = counter.process(frame)

        cv2.putText(results.plot_im, f"Count: {len(counter.counted_ids)}", (30, 50), # type: ignore
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3) # type: ignore

        cv2.imshow("Counting", results.plot_im)#type:ignore
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
