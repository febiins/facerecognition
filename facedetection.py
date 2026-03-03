import cv2
from deepface import DeepFace
import os

# Load Haar Cascade for detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30)
    )

    for (x, y, w, h) in faces:

        # Crop detected face
        face = frame[y:y+h, x:x+w]

        # Save temporarily
        cv2.imwrite("temp.jpg", face)

        try:
            # Transfer Learning Recognition
            result = DeepFace.find(
                img_path="temp.jpg",
                db_path="dataset/",
                model_name="Facenet",
                enforce_detection=False
            )

            if len(result) > 0 and not result[0].empty:
                identity_path = result[0].iloc[0]["identity"]
                person_name = os.path.basename(os.path.dirname(identity_path))
            else:
                person_name = "Unknown"

        except:
            person_name = "Unknown"

        # Draw rectangle
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        # Display name
        cv2.putText(frame, person_name,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0,255,0),
                    2)

    cv2.imshow("Face Recognition - Transfer Learning", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()