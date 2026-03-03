import cv2
import os

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ask for user name (better than numeric ID)
user_name = input("Enter Person Name: ")

# Create dataset folder structure
dataset_directory = "dataset"
person_folder = os.path.join(dataset_directory, user_name)

if not os.path.exists(person_folder):
    os.makedirs(person_folder)

# Start camera
cap = cv2.VideoCapture(0)

print("\n[INFO] Starting face capture...")
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        count += 1

        # Crop face
        face = frame[y:y+h, x:x+w]

        # Save image
        file_path = os.path.join(person_folder, f"{count}.jpg")
        cv2.imwrite(file_path, face)

        # Draw rectangle
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Capturing Faces", frame)

    # Stop after 50 images or press q
    if cv2.waitKey(100) & 0xFF == ord('q') or count >= 50:
        break

print(f"\n[INFO] Captured {count} images. Exiting.")

cap.release()
cv2.destroyAllWindows()