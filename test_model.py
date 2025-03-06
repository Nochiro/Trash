from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO(r'runs/detect/train5/weights/best.pt')

# Test on a single image
results = model.predict(
    source=r'C:\Users\Ramesh\OneDrive\Desktop\Trash detection\img1.jpg',
    save=True,
    conf=0.25
)

# OpenCV to keep the window open
img = cv2.imread(r'runs/detect/predict/img1.jpg')
cv2.imshow('Prediction', img)
cv2.waitKey(0)  # Wait until a key is pressed
cv2.destroyAllWindows()


