import cv2
from ultralytics import YOLO
import cvzone
import math
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("C:/Users/HP/PycharmProjects/posemodule/sample.mp4")
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("yolov8l.pt")
model2 = YOLO("yolov8n-seg.pt")
names = model.names
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    img2 = img.copy()
    for i in results:
        boxes = i.boxes
        for box in boxes:
            # Classes id = 0 -> person
            cls = box.cls[0]
            if int(cls) == 0:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                # x1, y1, w, h = box.xywh[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2-x1, y2-y1
                bbox = int(x1), int(y1), int(w), int(h)

                print(x1, y1, x2, y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Confidence
                conf = math.ceil((box.conf[0]*100))/100
                print(conf)

                res = model2(img)

    cv2.imshow("original", img2)
    cv2.imshow("multi-person segmented", img)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
