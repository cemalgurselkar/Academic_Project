import cv2
from ultralytics import YOLO
import math

class BatchCorrection:
    #Balıkların grup halinde (batch) geçişlerini analiz eden ve
    #hatalı sayımları engellemek için histogram tabanlı doğrulama yapan sınıf.

    # Nasıl Çalışıyor
    #Her frame'de tespit edilen balık sayısını analiz eder ve bir histogram (buffer) oluşturur.   
    #Mantık:
        #1. Veri Toplama (Buffer): Balıklar görüldüğü sürece, her frame'deki sayı bir sözlüğe kaydedilir 
        #   (Örn: {3 balık: 40 frame, 2 balık: 5 frame}). Bu, anlık hataları sönümlemeyi sağlar.
        
        #2. Dinamik Eşik (Dynamic Threshold): Balık grubu ekrandan çıktığında, grubun geçiş süresine 
        #   bakılarak bir güvenilirlik eşiği (%10 veya %15) belirlenir.
        
        #3. Karar Verme: Histogramdaki en yüksek sayıdan başlayarak, belirlenen eşiği geçen 
        #   ilk sayı 'kararlı sayım' olarak kabul edilir ve ilgili kasaya atanır.
        
    #Args:
    #    frame_id (int): Mevcut kare numarası.
    #    current_fish (list): O anki karedeki balık tespitleri.
    #    tracked_boxes (dict): Takip edilen kasaların listesi.
    #    
    #Returns:
    #    tuple: (kararli_sayi, hedef_kasa_id, durum_mesaji, debug_bilgileri)
        
    def __init__(self):
        self.count_histogram = {} 
        self.active = False
        
        # Süre Analizi
        self.total_batch_frames = 0 
        self.start_frame = 0
        self.last_exit_pos = None 

        # Cooldown ve Eşik
        self.patience = 8 
        self.empty_frame_count = 0 
        
        # Log dosyasını sıfırla
        with open("batch_log.txt", "w") as f:
            pass

    def get_dynamic_ratio(self, total_frames):
        #Frame süresine göre dinamik oran belirler.
        #- 30 frameden düşükse threshold: %15
        #- 30 frameden yüksek ise threshold: %10
        if total_frames < 30:
            return 0.15
        else:
            return 0.10

    def update(self, frame_id, current_fish, tracked_boxes):
        """
        Dönüş: (kararli_sayi, eklenecek_kutu_id, durum_mesaji, debug_lines_list)
        """
        current_fish_count = len(current_fish)
        status_message = ""
        debug_lines = [] # Ekrana basılacak satırları burada tutuyorum

        #1. durum: Balık Görüldü
        if current_fish_count > 0:
            self.empty_frame_count = 0 
            
            if not self.active:
                self.active = True
                self.start_frame = frame_id
                self.count_histogram = {} 
                self.total_batch_frames = 0
                self.last_exit_pos = None
            
            # Histogramı Doldur
            self.count_histogram[current_fish_count] = self.count_histogram.get(current_fish_count, 0) + 1
            self.total_batch_frames += 1

            # Konum Takibi, balığın ağırlık merkezini alıyoruz. En son konumda iken, hangi kasaya daha yakın onu hesaplamak için
            max_y = -1
            target_x = 0
            for f in current_fish:
                bbox = f[0]
                off_x, off_y = f[3]


                y_center = (bbox[1] + bbox[3]) // 2 + off_y 
                x_center = (bbox[0] + bbox[2]) // 2 + off_x

                if y_center > max_y:
                    max_y = y_center
                    target_x = x_center
            
            if max_y != -1:
                self.last_exit_pos = (target_x, max_y)

            sorted_display = dict(sorted(self.count_histogram.items(), reverse=True))
            status_message = f"Wait({self.total_batch_frames}f) | Buffer: {sorted_display}"
            
            return 0, None, status_message, []

        #2. durum: Balık Yok, Bekleme Modu
        elif self.active and current_fish_count == 0:
            self.empty_frame_count += 1
            sorted_display = dict(sorted(self.count_histogram.items(), reverse=True))

            if self.empty_frame_count < self.patience:
                return 0, None, f"Exit Wait ({self.empty_frame_count}/{self.patience}) | {sorted_display}", []

            #patience doldu ---
            self.active = False
            self.empty_frame_count = 0
            
            # debug raporu hazırlama
            debug_lines.append(f"--- Desicion (Frame {frame_id}) ---")
            debug_lines.append(f"Sure: {self.total_batch_frames} Frames")
            
            # Dinamik Oran Hesabı
            ratio = self.get_dynamic_ratio(self.total_batch_frames)
            threshold_val = self.total_batch_frames * ratio
            
            debug_lines.append(f"Ratio: %{int(ratio*100)} | Threshold: {threshold_val:.2f}")

            sorted_counts = sorted(self.count_histogram.items(), key=lambda item: item[0], reverse=True)
            
            final_count = 0
            found_valid = False
            
            # Eşik Kontrolü
            for count_val, frequency in sorted_counts:
                passed = frequency >= threshold_val
                status_str = "[OK]" if passed else "[FAIL]"
                
                # Ekrana sığması için kısa format: "3 Balik (12) -> [OK]"
                debug_lines.append(f"Count {count_val} (Freq:{frequency}) -> {status_str}")
                
                if passed:
                    final_count = count_val
                    found_valid = True
                    break 
            
            if not found_valid:
                debug_lines.append("RESULT: Count=0")

            with open("batch_log.txt", "a") as f:
                f.write(f"Frame_ID: {frame_id} -> {dict(sorted(self.count_histogram.items(), reverse=True))}, sayilan: {final_count}\n")

            # Kutu Ataması
            best_box_id = None
            min_distance = float('inf')

            if final_count > 0:
                if self.last_exit_pos and tracked_boxes:
                    exit_x, exit_y = self.last_exit_pos
                    
                    for bid, box in tracked_boxes.items():
                        box_bottom_x = (box.x1 + box.x2) // 2
                        box_bottom_y = box.y2 
                        dist = math.sqrt((exit_x - box_bottom_x)**2 + (exit_y - box_bottom_y)**2)
                        
                        if dist < min_distance:
                            min_distance = dist
                            best_box_id = bid
                    
                    if best_box_id is not None:
                        debug_lines.append(f"Target Box: {best_box_id} (Distance: {int(min_distance)})")
                    else:
                        debug_lines.append("Error: The box cannot be found")
                else:
                    debug_lines.append("WARNING: No Exit Position")

            result_msg = f"Done: Count {final_count} (Freq: {self.count_histogram.get(final_count, 0)})"
            self.count_histogram = {}
            
            if best_box_id is not None and final_count > 0:
                return final_count, best_box_id, result_msg, debug_lines
            
            return 0, None, result_msg, debug_lines
            
        return 0, None, "Idle - Waiting for Fish...", []


class Box:
    def __init__(self, box_id, xyxy, conf):
        self.box_id = box_id
        self.x1, self.y1, self.x2, self.y2 = map(int, xyxy)
        self.conf = conf
        self.missing_frames = 0

    def draw_box(self, frame):
        cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), (255, 0, 0), 2)
        bottom_center_x = (self.x1 + self.x2) // 2
        bottom_center_y = self.y2
        cv2.circle(frame, (bottom_center_x, bottom_center_y), 6, (0, 0, 255), -1)
        cv2.putText(frame, f"Box {self.box_id}", (self.x1, self.y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def update_box(self, xyxy, conf):
        self.x1, self.y1, self.x2, self.y2 = map(int, xyxy)
        self.conf = conf

class Fish:
    def __init__(self, fish_id, xyxy, conf, roi_offset):
        self.fish_id = fish_id
        self.conf = conf
        self.missed_frames = 0
        self.max_missed = 20
        self.track_history = []
        self.max_history = 30
        self.update_from_xyxy(xyxy, roi_offset)

    def update_from_xyxy(self, xyxy, roi_offset):
        fx1, fy1, fx2, fy2 = xyxy.astype(int)
        ox, oy  = roi_offset
        new_fx1, new_fy1 = fx1 + ox, fy1 + oy
        new_fx2, new_fy2 = fx2 + ox, fy2 + oy
        new_center = ((new_fx1 + new_fx2) // 2, (new_fy1 + new_fy2) // 2)
        
        self.track_history.append(new_center)
        if len(self.track_history) > self.max_history: self.track_history.pop(0)

        self.fx1, self.fy1 = new_fx1, new_fy1
        self.fx2, self.fy2 = new_fx2, new_fy2
        self.center = new_center

    def draw_fish(self, frame):
        color = (0, 255, 0)
        cv2.rectangle(frame, (self.fx1, self.fy1), (self.fx2, self.fy2), color, 2)
        cv2.circle(frame, self.center, 4, color, -1)

class DraggableROI:
    #Kullanıcının sabit boyutlu bir ROI (İlgi Alanı) kutusunu mouse ile tutup sürükleyerek
    #ekranda istediği konuma dinamik olarak yerleştirmesini ve seçimi onaylamasını sağlayan sınıf.
    #Eğer ROI boyutlarını değiştirmek istiyorsan, Detection classındaki, _select_roi_draggable fonksiyonundan yapabilirsin.
    def __init__(self, window_name, initial_rect):
        self.window_name = window_name
        self.x, self.y, self.w, self.h = initial_rect
        self.dragging = False
        self.offset_x = 0
        self.offset_y = 0

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.x < x < self.x + self.w and self.y < y < self.y + self.h:
                self.dragging = True
                self.offset_x = x - self.x
                self.offset_y = y - self.y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self.x = x - self.offset_x
                self.y = y - self.offset_y

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def select(self, cap):
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            display_frame = frame.copy()
            cv2.rectangle(display_frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)
            cv2.putText(display_frame, "Drag the ROI and press enter to work", (self.x, self.y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            h_img, w_img = frame.shape[:2]
            if self.x < 0: self.x = 0
            if self.y < 0: self.y = 0
            if self.x + self.w > w_img: self.x = w_img - self.w
            if self.y + self.h > h_img: self.y = h_img - self.h

            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(20) & 0xFF
            
            if key == 13 or key == 32: # Enter or Space
                break
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cv2.setMouseCallback(self.window_name, lambda *args: None)
        return (self.x, self.y, self.w, self.h)


class Detection:
    def __init__(self, source_video):
        self.box_model = YOLO('best-kasa.pt')
        self.fish_model = YOLO('best2.pt')
        self.source_video = source_video
        self.window_name = 'Fish_Tracking_Monitor'

        self.box_threshold = 0.9
        self.fish_threshold = 0.65

        self.tracked_box = {}
        self.max_missing_frame = 20
        self.tracked_fish = {}
        self.max_fish_miss = 35
        self.roi = None
        self.box_counter = {}

        self.batch_corrector = BatchCorrection()
        self.video_writer = None
        
        # Debug Overlay Değişkenleri
        self.debug_overlay_lines = []
        self.debug_timer = 0

    def _init_video(self):
        self.cap = cv2.VideoCapture(self.source_video)
        if not self.cap.isOpened(): raise ValueError('Video cannot be open!')
        cv2.namedWindow(self.window_name)
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_writer = cv2.VideoWriter("output.mp4", 
                                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    def _capture_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def _select_roi_draggable(self):
        initial_roi = (631, 344, 744, 344)
        # Eğer roinin boyutunu değiştirmek için buraya , buradaki initial_roi değiştirilmeli
        draggable = DraggableROI(self.window_name, initial_roi)
        self.roi = draggable.select(self.cap)
        if self.roi:
            print(f"ROI Sabitlendi: {self.roi}")
        return self.roi

    def _detect_boxes(self, frame):
        res = self.box_model.track(frame, persist=True, tracker='bytetrack.yaml', verbose=False)
        if not res or res[0].boxes.id is None: return []
        boxes = res[0].boxes
        return [(b, i, c) for b, i, c in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy(), boxes.conf.cpu().numpy()) if c >= self.box_threshold]

    def _detect_fish(self, frame, roi):
        if not roi or roi[2] == 0: return []
        x, y, w, h = roi
        safe_y = max(0, y)
        safe_h = min(frame.shape[0] - safe_y, h)
        safe_x = max(0, x)
        safe_w = min(frame.shape[1] - safe_x, w)

        crop = frame[safe_y:safe_y+safe_h, safe_x:safe_x+safe_w]
        if crop.size == 0: return []

        res = self.fish_model.track(crop, persist=True, tracker='botsort.yaml', verbose=False, iou=0.7)
        if not res or res[0].boxes.id is None: return []
        boxes = res[0].boxes
        return [(b, i, c, (safe_x, safe_y)) for b, i, c in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy().astype(int), boxes.conf.cpu().numpy()) if c >= self.fish_threshold]

    def _update_tracking(self, detections):
        curr_ids = {bid for _, bid, _ in detections}
        for box_xy, bid, conf in detections:
            if bid in self.tracked_box:
                self.tracked_box[bid].update_box(box_xy, conf)
                self.tracked_box[bid].missing_frames = 0
            else:
                self.tracked_box[bid] = Box(bid, box_xy, conf)
        
        to_remove = []
        for bid, box in self.tracked_box.items():
            if bid not in curr_ids:
                box.missing_frames += 1
                if box.missing_frames > self.max_missing_frame: to_remove.append(bid)
        for bid in to_remove: del self.tracked_box[bid]

    def _update_fish_tracking(self, detections):
        curr_ids = {fid for _, fid, _, _ in detections}
        for fxy, fid, conf, off in detections:
            if fid in self.tracked_fish:
                self.tracked_fish[fid].update_from_xyxy(fxy, off)
                self.tracked_fish[fid].missed_frames = 0
            else:
                self.tracked_fish[fid] = Fish(fid, fxy, conf, off)
        
        to_remove = []
        for fid, fish in self.tracked_fish.items():
            if fid not in curr_ids:
                fish.missed_frames += 1
                if fish.missed_frames > self.max_fish_miss: to_remove.append(fid)
        for fid in to_remove: del self.tracked_fish[fid]

    def _draw_debug_overlay(self, frame):
        #Sağ üst köşeye debug bilgilerini çizer.
        #Oradan her framede tespit edilen, sayılan balık bilgileri için debug olarak yaptım.
        if self.debug_timer > 0 and self.debug_overlay_lines:
            self.debug_timer -= 1
            
            x_start = frame.shape[1] - 350 
            y_start = 20
            line_height = 25
            
            overlay = frame.copy()
            box_h = len(self.debug_overlay_lines) * line_height + 20
            cv2.rectangle(overlay, (x_start - 10, y_start - 10), (frame.shape[1], y_start + box_h), (0, 0, 0), -1)
            
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            for i, line in enumerate(self.debug_overlay_lines):
                y = y_start + (i + 1) * line_height
                
                color = (255, 255, 255)
                if "[FAIL]" in line or "HATA" in line or "Yetersiz" in line:
                    color = (0, 0, 255)
                elif "[OK]" in line or "Hedef" in line:
                    color = (0, 255, 0)
                elif "Ratio" in line:
                    color = (0, 255, 255)

                cv2.putText(frame, line, (x_start, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    def run(self):
        self._init_video()
        roi = self._select_roi_draggable()
        if not roi or roi[2] == 0: return
        x, y, w, h = roi
        
        frame_id = 0
        current_status_msg = "Ready"

        while True:
            frame = self._capture_frame()
            if frame is None: break
            frame_id += 1 

            self._update_tracking(self._detect_boxes(frame))
            curr_fish = self._detect_fish(frame, roi)
            self._update_fish_tracking(curr_fish)

            count_to_add, target_box, msg, debug_lines = self.batch_corrector.update(frame_id, curr_fish, self.tracked_box)

            if debug_lines:
                self.debug_overlay_lines = debug_lines
                self.debug_timer = 150 

            if msg: current_status_msg = msg

            if count_to_add > 0:
                if target_box not in self.box_counter: self.box_counter[target_box] = 0
                self.box_counter[target_box] += count_to_add

                cv2.putText(frame, f"+{count_to_add}", (x + w - 50, y + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, current_status_msg, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) 
            
            for box in self.tracked_box.values(): box.draw_box(frame)
            for fish in self.tracked_fish.values(): fish.draw_fish(frame)

            self._draw_debug_overlay(frame)

            idx = 0
            for bid in sorted(self.box_counter.keys()):
                count = self.box_counter[bid]
                if count > 0:
                    cv2.putText(frame, f"Kasa {int(bid)}: {count}", (30, 50 + idx * 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    idx += 1

            cv2.imshow(self.window_name, frame)
            if self.video_writer: self.video_writer.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
        
        with open("fish_counts_monitor4.txt", 'w') as f:
            f.write("Fish Counting Report\n")
            f.write("Kutu ID\tSayi\n")
            for bid in sorted(self.box_counter.keys()):
                f.write(f"{int(bid)}\t{self.box_counter[bid]}\n")
        
        print("Sonuçlar kaydedildi.")
        if self.video_writer: self.video_writer.release()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = "/path/your_video.mp4"
    Detection(video_path).run()