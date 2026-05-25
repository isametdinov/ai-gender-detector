import cv2
from ultralytics import YOLO

# Путь к твоей обученной модели классификации
MODEL_PATH = "gender_yolo11n_v2.pt"

# Инициализация модели классификации
model = YOLO(MODEL_PATH)

# Захват видео (0 - веб-камера)
cap = cv2.VideoCapture(0)

print("Запуск классификации. Нажмите 'q' для выхода.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Классификация кадра
    # verbose=False отключает вывод каждого кадра в консоль
    results = model.predict(frame, verbose=False)

    # Получаем результат
    result = results[0]
    
    # Имя класса с самой высокой вероятностью
    top1_idx = result.probs.top1
    top1_name = result.names[top1_idx]
    confidence = result.probs.top1conf.item()

    # Отображение на кадре
    label = f"{top1_name} ({confidence:.2f})"
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Gender Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
