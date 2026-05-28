import cv2
from ultralytics import YOLO
import yt_dlp
import os
import sqlite3

# Пути к моделям и БД
DETECTOR_PATH = "yolo11n.pt"
CLASSIFIER_PATH = "gender_yolo11n_v2.pt"
YOUTUBE_URL = "https://youtu.be/8gxA-5-VWI0?si=E8LoqIYzcyu4mnDP"
DB_PATH = "cv_analytics.db"

# Настройки
ROI = [200, 140, 360, 160] # X, Y, W, H
DRAW_ROI = True
DRAW_BOXES = True

# Локальный кэш для БД
processed_ids = set()

# Загрузка моделей
detector = YOLO(DETECTOR_PATH)
classifier = YOLO(CLASSIFIER_PATH)

# Конфигурация YT-DLP
ydl_opts = {
    'format': 'best[ext=mp4]/best',
    'cookiefile': 'cookies.txt',
    'verbose': False,
    'js_runtimes': {'node': {}},
    'enable_remote_components': True,
    'compat_opts': ['no-youtube-unavailable-videos', 'all']
}

print("Запуск системы. Нажмите 'q' для выхода.")

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
                cv2.rectangle(frame, (ROI[0], ROI[1]), (ROI[0] + ROI[2], ROI[1] + ROI[3]), (255, 0, 0), 1)

            results = detector.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, tid in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    
                    if (x1 >= ROI[0] and y1 >= ROI[1] and 
                        x2 <= ROI[0] + ROI[2] and y2 <= ROI[1] + ROI[3]):
                        
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0: continue
                        
                        cls_results = classifier.predict(crop, verbose=False)
                        top1_name = cls_results[0].names[cls_results[0].probs.top1]
                        conf = cls_results[0].probs.top1conf.item()

                        # Логирование в БД (processed_ids уже глобален)
                        if tid not in processed_ids:
                            conn = sqlite3.connect(DB_PATH)
                            cursor = conn.cursor()
                            cursor.execute("INSERT INTO people_tracks (track_id, label) VALUES (?, ?)", (int(tid), top1_name))
                            conn.commit()
                            conn.close()
                            processed_ids.add(tid)
                            print(f"[БАЗА ДАННЫХ] Сохранено: ID {tid} | Класс: {top1_name}")

                        if DRAW_BOXES:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                            cv2.putText(frame, f"{top1_name} {conf:.2f}", (x1, y1 - 5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            cv2.imshow("Main Inference", cv2.resize(frame, (960, 540)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                exit()
        cap.release()
    except Exception as e:
        print(f"Ошибка: {e}. Переподключение через 10 секунд...")
        cv2.waitKey(10000)
