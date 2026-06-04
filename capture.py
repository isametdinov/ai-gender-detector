import cv2
import sqlite3
from ultralytics import YOLO
import yt_dlp
import os

# 1. Настройки путей и линков
MODEL_PATH = "gender_yolo11.pt"
YOUTUBE_URL = "https://www.youtube.com/watch?v=BAw342Xqxhs&pp=ygUMY2l0eSBzdHJlZXRz"
DB_PATH = "cv_analytics.db"
ROI_LIMITS = None  # Отключено, анализируем весь кадр целиком

# Опции для стабильного чтения сетевого стрима через FFMPEG
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"

# 2. ПОДКЛЮЧАЕМСЯ К ЛОКАЛЬНОЙ КАМЕРЕ
print("Подключаемся к веб-камере...")
cap = cv2.VideoCapture(0)

# 3. Инициализация YOLO
model = YOLO(MODEL_PATH)

# Настройка (анализируем каждый кадр для теста)
frame_skip_interval = 1
frame_count = 0

print("Запуск анализа камеры. Нажмите 'q' для выхода.\n" + "-" * 50)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ошибка: Не удалось получить кадр со стрима.")
        break

    frame_count += 1

    # Пропуск кадров (показываем оригинальный кадр без детекции)
    if frame_count % frame_skip_interval != 0:
        cv2.imshow("CV Project Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # Подготовка кадра (ROI отключен, берем фрейм целиком)
    roi_frame = frame

    # Запускаем трекер
    results = model.track(roi_frame, persist=True, tracker="bytetrack.yaml", conf=0.15, verbose=True)

    # Проверяем, есть ли люди в кадре
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        for box, track_id, cls_id in zip(boxes, track_ids, clss):
            class_name = model.names[cls_id]

            # ВЫВОД ЛОГА В ТЕРМИНАЛ
            print(f"[ДЕТЕКЦИЯ] Кадр: {frame_count} | ID: {track_id} | Класс: {class_name}")

            x1, y1, x2, y2 = map(int, box)

            # Выбор цвета бокса (B, G, R)
            color = (255, 0, 0)  # Синий для Men
            if class_name == 'woman':
                color = (203, 192, 255)  # Розовый
            elif class_name == 'child':
                color = (0, 255, 0)  # Зеленый

            # Рисуем рамку вокруг человека
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Формируем и рисуем надпись с подложкой
            label = f"ID: {track_id} | {class_name.upper()}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Запись в базу данных SQLite
            cursor.execute("""
                           INSERT
                           OR IGNORE INTO people_tracks (track_id, gender_age) 
                VALUES (?, ?)
                           """, (int(track_id), class_name))

        conn.commit()
        conn.close()
    else:
        # Лог, если кадр проверен, но людей нет
        print(f"Кадр {frame_count}: В зоне видимости никого нет.")

    # Выводим итоговый кадр с боксами
    cv2.imshow("CV Project Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("-" * 50 + "\nАнализ потока завершен.")