import cv2 #import the open cv 

#loading the pre-trained model into a variable using cascade classifier 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

cap = cv2.VideoCapture(0) #using the webcam to capture the video

print(" Web cam started. Press 'q' to quit")

while True:
    ret,frame = cap.read() #reading the frame from the webcam the ret holds the boolean value and frame holds the image array
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converting the image to gray scale
    faces =face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30)) #detecting the faces in the image
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #drawing the rectangle around the face
    cv2.imshow("Face Detection", frame) #displaying the frame with the detected faces
    if cv2.waitKey(1) & 0xFF == ord('q'):#waiting for the user to press 'q'
        break

cap.release()
cv2.destroyAllWindows()