import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# Choose your video source:
# 0 for default webcam
# 'path/to/your/video.mp4' for a video file
video_source = 0 # or 'path/to/your/video.mp4'

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"Error: Could not open video source {video_source}")
    exit()

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Perform inference on the current frame
        # The 'stream=True' argument makes it suitable for real-time processing
        results = model.predict(source=frame, conf=0.5, stream=True)

        # Iterate over the results for the current frame
        for r in results:
            # Get the annotated frame (with bounding boxes and labels)
            annotated_frame = r.plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Real-time Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the video has ended or an error occurred
        break

# Release the video capture object and close display windows
cap.release()
cv2.destroyAllWindows()
print("Real-time detection stopped.")

