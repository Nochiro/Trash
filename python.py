from ultralytics import YOLO

if __name__ == '__main__':
    # Initialize a new YOLOv8 model (no pre-trained weights)
    model = YOLO('yolov8n.yaml')  # Using model architecture only, no weights

    # Train the model from scratch on GPU
    results = model.train(
        data=r'C:\Users\23101B0062\Downloads\Trash-main\Trash-main\data.yaml',
        epochs=50,
        batch=16,
        imgsz=640,
        device='cuda',  # Runs on RTX 4090
        workers=4,
        pretrained=False  # Ensures no pre-trained weights are loaded
    )

    # Evaluate the model
    metrics = model.val()

    # Save/export the trained model (optional)
    model.export(format='onnx')





