from time import sleep

import cv2

import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array

face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier =load_model(r'Emotion_little_vgg.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
def facedetector(src):
    cap = cv2.imread(src,1)
    img = cv2.resize(cap, (480, 480))

    def detection(img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        # rect,face,image = face_detector(img)
            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

            # make a prediction on the ROI, then lookup the class

                preds = classifier.predict(roi)[0]
                label=class_labels[preds.argmax()]
                label_position = (x,y)
                cv2.putText(img,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            else:
                cv2.putText(img,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        cv2.imshow('Emotion Detector',img)
    detection(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
