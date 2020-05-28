import numpy as np
import cv2
from keras.models import load_model

# loading the best model
model = load_model('/home/sadabrata/transfer_learning_opencv/best_model.h5', compile = False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# face classifier
face_cascade = cv2.CascadeClassifier('/home/sadabrata/transfer_learning_opencv/files/haarcascade_frontalface_default.xml')

# Function detects faces and returns the cropped face
def face_extractor(img):
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if faces is ():
        return None
    # crop all faces found
    for (x,y,w,h) in faces:
        # draw rectangle on face
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        # crop the face
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face

# using webcam to capture video
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    face = face_extractor(frame)
    #print("face datatype",type(face))
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        face_arr = np.expand_dims(face, axis=0)
        pred = model.predict(face_arr)
        print(pred)
        if(pred[0][0]>0.60):
            name="Nilanjan "+str(round(100*pred[0][0],2))+"%"
            cv2.putText(frame,name, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        elif(pred[0][1]>0.60):
            name="Sadabrata "+str(round(100*pred[0][1],2))+"%"
            cv2.putText(frame,name, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        elif(pred[0][2]>0.60):
            name="Soumyadeep "+str(round(100*pred[0][2],2))+"%"
            cv2.putText(frame,name, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(frame,"No face found, face the camera", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
