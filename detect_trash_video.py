import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO(r'runs/detect/train5/weights/best.pt')

# Open the video file
video_path = r'C:\Users\Ramesh\OneDrive\Desktop\Trash detection\video1.mp4'
cap = cv2.VideoCapture(video_path)

# Get video details
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
output_path = 'path/to/output/video.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model.predict(frame)

    # Visualize results on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('Trash Detection', annotated_frame)

    # Write the frame to the output video
    out.write(annotated_frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
