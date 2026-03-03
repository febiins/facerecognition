import cv2
import numpy as np
from deepface import DeepFace
import os

# -------------------------------
# Check if embeddings exist
# -------------------------------
if not os.path.exists("embeddings.npy") or not os.path.exists("labels.npy"):
    print("Error: embeddings.npy or labels.npy not found.")
    print("Run transfer_training.py first.")
    exit()

# Load trained embeddings
stored_embeddings = np.load("embeddings.npy")
stored_labels = np.load("labels.npy")

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# FIXED CAMERA BACKEND (Windows)
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting Face Recognition using Transfer Learning...")
print("Press 'q' to exit.")

while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:

        # Draw rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        face = img[y:y+h, x:x+w]
        cv2.imwrite("temp.jpg", face)

        try:
            # Extract embedding using FaceNet
            result = DeepFace.represent(
                img_path="temp.jpg",
                model_name="Facenet",
                detector_backend="opencv",
                enforce_detection=False
            )

            new_embedding = np.array(result[0]["embedding"])

            # Calculate Euclidean distance
            distances = np.linalg.norm(stored_embeddings - new_embedding, axis=1)
            min_index = np.argmin(distances)
            min_distance = distances[min_index]

            # Threshold tuning (important)
            if min_distance < 10:
                name = stored_labels[min_index]
            else:
                name = "Unknown"

        except Exception as e:
            name = "Unknown"

        # Display name
        cv2.putText(img, name,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)

    cv2.imshow("Transfer Learning Face Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()