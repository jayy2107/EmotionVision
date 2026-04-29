# ============================================================
# Project  : Facial Emotion Recognition
# Made by  : JAYENDRA SINGH
# College  : [Your College Name]
# Tech     : Python, OpenCV, DeepFace, TensorFlow
# ============================================================

# ---------- IMPORTING LIBRARIES ----------

import cv2                          # OpenCV - used for camera and image processing
from deepface import DeepFace       # DeepFace - AI library for emotion detection

# ---------- LOADING FACE DETECTOR ----------

# Haar Cascade is a pre-trained model that knows how to find faces in images
# It uses the XML file we downloaded which contains thousands of face patterns
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ---------- STARTING THE WEBCAM ----------

# cv2.VideoCapture(0) opens the default webcam (built-in camera on MacBook)
# 0 = first camera, 1 = second camera (if connected)
cap = cv2.VideoCapture(0)

print("Camera started. Press 'Q' to quit.")

# ---------- MAIN LOOP ----------

# This loop runs continuously - capturing and processing frames one by one
while True:

    # Read one frame from the webcam
    # ret  = True if frame was captured successfully
    # frame = the actual image/photo captured
    ret, frame = cap.read()

    # If frame was not captured properly, skip and try again
    if not ret:
        print("Failed to grab frame")
        break

    # ---------- CONVERT TO GRAYSCALE ----------

    # Grayscale (black & white) is easier and faster to process
    # Face detection works better on grayscale images
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------- DETECT FACES ----------

    # detectMultiScale scans the image and finds all faces
    # scaleFactor=1.1   → how much the image is scaled each time (to find faces at different sizes)
    # minNeighbors=5    → how many neighbors each face needs to be considered a real face
    # minSize=(30, 30)  → smallest face size to detect (in pixels)
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # ---------- PROCESS EACH DETECTED FACE ----------

    # Loop through each detected face
    # x, y = top-left corner of the face box
    # w, h = width and height of the face box
    for (x, y, w, h) in faces:

        # Crop just the face area from the original frame
        face_roi = frame[y:y + h, x:x + w]

        try:
            # ---------- DETECT EMOTION USING DEEPFACE ----------

            # DeepFace analyzes the face and predicts the emotion
            # actions=['emotion'] tells it to only check for emotions
            # enforce_detection=False prevents crash if face is not perfect
            result = DeepFace.analyze(
                face_roi,
                actions=['emotion'],
                enforce_detection=False
            )

            # Get the dominant (strongest) emotion from the result
            # Example: 'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral'
            emotion = result[0]['dominant_emotion']

        except Exception as e:
            # If any error happens during detection, just show "Unknown"
            emotion = "Unknown"

        # ---------- DRAW RECTANGLE AROUND FACE ----------

        # Draw a green rectangle box around the detected face
        # (x, y) = top-left corner
        # (x+w, y+h) = bottom-right corner
        # (0, 255, 0) = green color in BGR format
        # 2 = thickness of the border line
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ---------- SHOW EMOTION LABEL ON SCREEN ----------

        # Write the emotion text above the face rectangle
        # cv2.FONT_HERSHEY_SIMPLEX = font style
        # 0.9 = font size
        # (0, 255, 0) = green color
        # 2 = thickness
        cv2.putText(
            frame,
            emotion,           # Text to display (e.g., "happy")
            (x, y - 10),       # Position: just above the rectangle
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    # ---------- DISPLAY THE FRAME ----------

    # Show the processed frame with rectangles and labels on screen
    cv2.imshow('JAYENDRA SINGH - Facial Emotion Recognition', frame)

    # ---------- CHECK FOR QUIT KEY ----------

    # Wait 1 millisecond for a key press
    # If the user presses 'Q' or 'q', break the loop and stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# ---------- CLEANUP ----------

# Release the webcam so other apps can use it
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print("Program ended successfully.")
