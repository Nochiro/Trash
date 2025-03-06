from ultralytics import YOLO
import cv2
import os

# Load the trained YOLOv8 model
model = YOLO(r'C:\Users\23101B0062\Downloads\Trash-main\Trash-main\runs\detect\train5\weights\best.pt')

# Path to the image
image_path = r'C:\Users\23101B0062\Downloads\Trash-main\Trash-main\img2.jpg'

# Run YOLOv8 inference on the image
results = model.predict(
    source=image_path,
    save=True,
    conf=0.25
)

# Get the latest prediction folder
runs_dir = r'C:\Users\23101B0062\Downloads\Trash-main\Trash-main\runs\detect'
latest_predict_folder = max([os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if d.startswith('predict')], key=os.path.getmtime)

# Path to the predicted image
predicted_image_path = os.path.join(latest_predict_folder, 'img2.jpg')

# Check if the predicted image exists and display it
if os.path.exists(predicted_image_path):
    img = cv2.imread(predicted_image_path)
    if img is not None:
        cv2.imshow('Trash Detection - Prediction', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: OpenCV could not load the image.")
else:
    print(f"Prediction image not found at: {predicted_image_path}")






