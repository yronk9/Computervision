import cv2
import pytesseract
import numpy as np
import re  # For regex
from datetime import datetime

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize the video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret will be True
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get a binary image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Use Tesseract to do OCR and get detailed data (including bounding boxes)
    data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)

    # Filter the recognized text to only include complete words (ignoring symbols)
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 60:  # Confidence level threshold for better accuracy
            word = data['text'][i]
            if re.match(r'\b[A-Za-z]+\b', word):  # Only match complete words
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
                cv2.putText(frame, word, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Get the current time with seconds
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Display the current time on the original frame
    cv2.putText(frame, f'Time: {current_time}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the "Press 'q' to exit" message
    cv2.putText(frame, "Press 'q' to exit", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Stack the original frame and preprocessed frame side by side
    combined_frame = np.hstack((frame, cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)))

    # Display the resulting frame with both the original and preprocessed images
    cv2.imshow('Original and Preprocessed Frames', combined_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture when done
cap.release()
cv2.destroyAllWindows()
