import cv2
from ultralytics import YOLO
import os

# Load the trained YOLOv8 model
model = YOLO(r'C:\Users\23101B0062\Downloads\Trash-main\Trash-main\runs\detect\train5\weights\best.pt')

# Open the video file
video_path = r'C:\Users\23101B0062\Downloads\Trash-main\Trash-main\video1.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

# Get video details
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer setup
output_dir = r'C:\Users\23101B0062\Downloads\Trash-main\Trash-main\runs\detect\output'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'detected_trash_output.mp4')

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or read error")
        break

    # Run YOLOv8 inference on the frame
    results = model.predict(frame, conf=0.25)

    # Visualize results on the frame
    annotated_frame = results[0].plot()

    # Display the frame with detections
    cv2.imshow('Trash Detection - Live', annotated_frame)

    # Write the frame with detections to the output video
    out.write(annotated_frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Video stopped by user")
        break

# Release video capture and writer resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Detection complete! Video saved to: {output_path}")
