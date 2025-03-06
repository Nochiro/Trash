from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # You can switch to 's', 'm', 'l', 'x' depending on power

# Train the model from scratch
results = model.train(
    data=r'C:\Users\Ramesh\OneDrive\Desktop\Trash detection\data.yaml',
    epochs=50,
    batch=8,
    imgsz=640,
    device='cpu'
)

# Evaluate the model
metrics = model.val()

# Save/export the trained model (optional)
model.export(format='onnx')

