import cv2 #import the open cv
import os #import the os module

dataset_directory = "dataset"
if not os.path.exists(dataset_directory):
    os.makedirs(dataset_directory)

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
# Get a unique ID for the user
user_id = input('Enter a numeric User ID and look at the camera: ')
print("\n[INFO] Starting face capture...")
count =0
while True:
    ret,frame=cap.read()
    if not ret:
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
    for(x,y,w,h) in faces:
        count+=1
        # Save the cropped face image to the dataset folder
        file_path = f"{dataset_directory}/User.{user_id}.{count}.jpg"
        cv2.imwrite(file_path, gray[y:y+h, x:x+w])
        cv2.imshow('Capturing Faces', frame)
        # Stop after 50 images or if 'q' is pressed
    if cv2.waitKey(100) & 0xFF == ord('q') or count >= 50:
        break
print(f"\n[INFO] Captured {count} images. Exiting.")
cap.release()
cv2.destroyAllWindows()