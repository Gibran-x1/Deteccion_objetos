import torch
import cv2
import numpy as np

# Lectura del modelo
model = torch.hub.load('ultralytics/yolov5','custom', path = 'D:/fcmdr/Documents/Programas/Python/Semestre6/ObjDet/pt/5Objects.pt')
#model = torch.hub.load('ultralytics/yolov5','custom', path = 'D:/fcmdr/Documents/Programas/Python/Semestre6/ObjDet/pt/Traffic.pt')

# Toma de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detecciones
    detect = model(frame)

    cv2.imshow('Detector', np.squeeze(detect.render()))

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()