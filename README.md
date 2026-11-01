# Workshop-2---Object-detection-using-web-camera

## AIM
To implement real-time object detection using the YOLOv8 model and a webcam. The system detects objects in video frames, draws bounding boxes around detected objects, and displays their labels with confidence scores.

## ALGORITHM

1. Import the required libraries: Ultralytics YOLO, OpenCV, Matplotlib, and IPython display.
2. Load the pretrained YOLOv8 medium model using `yolov8m.pt`.
3. Initialize the webcam using `cv2.VideoCapture(0)`.
4. Check whether the webcam is opened successfully.
5. Capture video frames continuously from the webcam.
6. Pass each frame to the YOLOv8 model for object detection with a confidence threshold of 0.60.
7. Obtain the detection results and draw bounding boxes and labels using the `plot()` function.
8. Convert the annotated frame from BGR to RGB format for proper display using Matplotlib.
9. Clear the previous output and display the updated frame with the title "Real-Time Object Detection".
10. Repeat the process until the video capture stops or a frame cannot be captured.
11. Release the webcam and close all OpenCV windows.

## PROGRAM

```python
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output

model = YOLO("yolov8m.pt")
cap = cv2.VideoCapture(0)

# Check if camera opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame.")
            break

        # Perform object detection
        results = model(
            frame,
            conf=0.60,
            verbose=False
        )

        # Draw bounding boxes and labels
        annotated_frame = results[0].plot()

        # Convert BGR to RGB for Matplotlib
        annotated_frame = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB
        )

        # Display in Jupyter
        clear_output(wait=True)
        plt.figure(figsize=(10, 8))
        plt.imshow(annotated_frame)
        plt.title("Real-Time Object Detection")
        plt.axis("off")
        plt.show()

# Release webcam
cap.release()
cv2.destroyAllWindows()
```
## OUTPUT
<img width="794" height="621" alt="download" src="https://github.com/user-attachments/assets/6a79f633-72e2-405b-8da3-960ad4cfdf43" />



## RESULT

Thus, real-time object detection was successfully implemented using the YOLOv8 medium model and webcam. The system detected objects in live video frames and displayed bounding boxes, object labels, and confidence scores for the detected objects.
