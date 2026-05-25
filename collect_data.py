import cv2
from ultralytics import YOLO
import yt_dlp
import os
import uuid

# 1. Настройки
YOUTUBE_URL = "https://www.youtube.com/watch?v=8JCk5M_xrBs"
OUTPUT_DIR = "dataset_raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ROI [x, y, w, h]
ROI = [680, 360, 1080, 540] 

# Настройки отрисовки
DRAW_ROI = False
DRAW_BOXES = False

# Инициализация детектора
detector = YOLO('gender_yolo11n_v2.pt')

# Конфигурация YT-DLP
ydl_opts = {
    'format': 'best[ext=mp4]/best',
    'cookiefile': 'cookies.txt',
    'verbose': False,
    'js_runtimes': {'node': {}},
    'enable_remote_components': True,
    'compat_opts': ['no-youtube-unavailable-videos', 'all']
}

seen_ids = set()
print("Сборщик запущен. Нажмите 'q' для выхода.")

while True:
    try:
        print("Получаем URL стрима через yt-dlp...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(YOUTUBE_URL, download=False)
            stream_url = info['url']
        
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            if DRAW_ROI:
                cv2.rectangle(frame, (ROI[0], ROI[1]), (ROI[0] + ROI[2], ROI[1] + ROI[3]), (255, 0, 0), 2)

            results = detector.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, tid in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    if DRAW_BOXES:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    if (x1 >= ROI[0] and y1 >= ROI[1] and 
                        x2 <= ROI[0] + ROI[2] and y2 <= ROI[1] + ROI[3]):
                        
                        if tid not in seen_ids:
                            h, w = frame.shape[:2]
                            x1_c = max(0, x1 - 10)
                            y1_c = max(0, y1 - 10)
                            x2_c = min(w, x2 + 10)
                            y2_c = min(h, y2 + 10)
                            crop = frame[y1_c:y2_c, x1_c:x2_c]
                            
                            file_name = f"id_{tid}_{uuid.uuid4().hex[:6]}_person.jpg"
                            cv2.imwrite(os.path.join(OUTPUT_DIR, file_name), crop)
                            seen_ids.add(tid)
                            print(f"Сохранен человек ID: {tid}")

            cv2.imshow("Data Collector", cv2.resize(frame, (960, 540)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                exit()
        cap.release()
    except Exception as e:
        print(f"Ошибка: {e}. Переподключение через 10 секунд...")
        cv2.waitKey(10000)
