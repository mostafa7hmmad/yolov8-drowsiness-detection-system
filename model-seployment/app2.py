import cv2
from ultralytics import YOLO
import pygame
import threading
import time

# تحميل الموديل
model = YOLO("best.pt")

# تعيين الكلاسات
CLASS_NAMES = {0: "microsleep", 1: "neutral", 2: "yawning"}
NATURAL_CLASS_ID = 1

# إعدادات البازر
pygame.mixer.init()
pygame.mixer.music.load("1.mp3")

# ثريد البازر
buzzer_on = False
buzzer_lock = threading.Lock()

def control_buzzer():
    global buzzer_on
    while True:
        with buzzer_lock:
            if buzzer_on:
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(-1)
            else:
                pygame.mixer.music.stop()
        time.sleep(0.1)

buzzer_thread = threading.Thread(target=control_buzzer, daemon=True)
buzzer_thread.start()

# وقت بداية ظهور abnormal class
abnormal_start_time = None
DELAY_SECONDS = 1  # المدة المطلوبة قبل تشغيل البازر

# فتح الكاميرا
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    boxes = results[0].boxes
    classes = boxes.cls.cpu().numpy() if boxes.cls is not None else []
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
    xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []

    frame_has_abnormal = False

    for i, (box, cls_id, conf) in enumerate(zip(xyxy, classes, confs)):
        x1, y1, x2, y2 = map(int, box)
        class_name = CLASS_NAMES.get(int(cls_id), "unknown")
        label = f"{class_name} ({conf:.2f})"

        if int(cls_id) == NATURAL_CLASS_ID:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
            frame_has_abnormal = True

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

    current_time = time.time()

    # إذا ظهر كلاس غلط
    if frame_has_abnormal:
        if abnormal_start_time is None:
            abnormal_start_time = current_time
        elif current_time - abnormal_start_time >= DELAY_SECONDS:
            with buzzer_lock:
                buzzer_on = True
    else:
        abnormal_start_time = None
        with buzzer_lock:
            buzzer_on = False

    cv2.imshow("YOLOv8 - Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
