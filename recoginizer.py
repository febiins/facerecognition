import cv2

recoginizer = cv2.face.LBPHFaceRecognizer_create()
recoginizer.read('trainer/trainer.yml')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font=cv2.FONT_HERSHEY_SIMPLEX

names = ['None', 'febin', 'febin']

cam = cv2.VideoCapture(0)

print("Starting to recognize...")
while True:
    ret, img = cam.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Predict the face
        id, confidence = recoginizer.predict(gray[y:y+h, x:x+w])

        # A lower confidence score is a better match (0 is perfect)
        if confidence < 100:
            # Check if the predicted ID is within our list range
            id_name = names[id] if id < len(names) else "Unknown"
            confidence_text = f" {round(100 - confidence)}%"
        else:
            id_name = "Unknown"
            confidence_text = ""
        
        # Display the name and confidence on the screen
        cv2.putText(img, str(id_name), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, confidence_text, (x+5, y+h-5), font, 1, (255, 255, 0), 1)

    cv2.imshow('Face Recognition', img)

    # Exit if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

print("\n[INFO] Exiting program.")
cam.release()
cv2.destroyAllWindows()

