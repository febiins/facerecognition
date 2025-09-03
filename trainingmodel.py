import cv2
import numpy as np
from PIL import Image
import os

path = "dataset"

recognizer = cv2.face.LBPHFaceRecognizer_create()

def getImagesandLables(path):
    imagepaths=[os.path.join(path,f)for f in os.listdir(path)]
    facesamples=[]
    ids=[]
    for imagepath in imagepaths:
        PIL_img=Image.open(imagepath).convert('L') #convert it to grayscale
        img_numpy=np.array(PIL_img,'uint8')
        id=int(os.path.split(imagepath)[-1].split(".")[1])
        facesamples.append(img_numpy)
        ids.append(id)
    return facesamples,ids

faces,ids=getImagesandLables(path)
recognizer.train(faces,np.array(ids))

trainer_directory = 'trainer'
if not os.path.exists(trainer_directory):
    os.makedirs(trainer_directory)
recognizer.save(trainer_directory + '/trainer.yml')
print(f"\n[INFO] Model trained for {len(np.unique(ids))} users. Exiting.")

