import cv2
import numpy as np
import time
import sys
from tensorflow.keras.models import load_model

# =====================
# VARIABLES
# =====================
width = 640
height = 480
th = 0.3   # seuil abaissé

# =====================
# LOAD MODEL
# =====================
model = load_model("mytrained.keras")

# =====================
# CAMERA
# =====================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No webcam found")
    sys.exit(1)

cap.set(3, width)
cap.set(4, height)

# =====================
# PREPROCESS
# =====================
def imgpreprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

# =====================
# LOOP
# =====================
while True:
    success, frame = cap.read()
    if not success:
        continue

    # ROI (zone verte)
    x1, y1, x2, y2 = 100, 100, 400, 400
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    img = cv2.resize(roi, (32,32))
    img = imgpreprocessing(img)
    img = img.reshape(1,32,32,1)

    preds = model.predict(img, verbose=0)
    classIndex = np.argmax(preds)
    confidence = np.max(preds)

    if confidence > th:
        text = f"{classIndex} ({confidence:.2f})"
        cv2.putText(frame, text, (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
