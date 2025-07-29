import cv2
import mediapipe as mp
from ultralytics import YOLO

# --- YOLOv8 Setup ---
# Load a pre-trained YOLOv8 model for general object detection (e.g., persons)
model = YOLO('yolov8n.pt') # You can choose yolov8s.pt, yolov8m.pt, etc.

# --- MediaPipe Hands Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, # True for images, False for video stream
    max_num_hands=2,         # Detect up to 2 hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Video Source Setup ---
# Choose your video source:
# 0 for default webcam
# 'path/to/your/video.mp4' for a video file
video_source = 0 # or 'path/to/your/video.mp4'

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"Error: Could not open video source {video_source}")
    exit()

print("Starting detection... Press 'q' to quit.")

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Flip the frame horizontally for a more natural webcam feel (optional)
        frame = cv2.flip(frame, 1)

        # --- YOLOv8 Inference (for persons) ---
        # Only detect 'person' if that's what you're interested in, or remove classes for all COCO objects
        # The COCO dataset class ID for 'person' is 0
        yolo_results = model.predict(source=frame, conf=0.5, classes=[0], verbose=False) # classes=[0] for person only

        # Get the annotated frame from YOLOv8 results (with person bounding boxes)
        annotated_frame = frame.copy() # Start with a clean copy for drawing
        for r in yolo_results:
            annotated_frame = r.plot() # Draws bounding boxes and labels for persons

        # --- MediaPipe Hands Inference ---
        # Convert the frame to RGB as MediaPipe expects RGB input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(frame_rgb)

        # Draw hand landmarks if hands are detected
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Display the combined annotated frame
        cv2.imshow("Person & Hand Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the video has ended or an error occurred
        break

# Release resources
cap.release()
hands.close() # Release MediaPipe resources
cv2.destroyAllWindows()
print("Detection process stopped.")