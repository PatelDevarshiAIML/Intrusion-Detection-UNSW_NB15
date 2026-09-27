import torch
from ultralytics import YOLO
import cv2
import pandas as pd

# Load the custom YOLOv8 model
import os`nmodel_path = os.path.join(os.path.dirname(__file__), "sign.pt")`nmodel = YOLO(model_path)

# Define class names corresponding to classes A to Y
class_names = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Define codec and create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Check if any objects are detected
    detections = results[0].boxes.data
    if len(detections) == 0:
        print("No objects detected.")
    else:
        # Iterate over each detection and draw bounding boxes with class names
        px = pd.DataFrame(detections).astype("float")
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            class_index = int(row[5])
            label = class_names.get(class_index, 'Unknown')  # Get class name from dictionary
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the frame to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if the job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
