import cv2

# face classifier
face_classifier = cv2.CascadeClassifier('/home/sadabrata/transfer_learning_opencv/files/haarcascade_frontalface_default.xml')

# detect face and return the cropped face
def face_extractor(img):
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    if faces is ():
        return None
    # crop all faces found
    for (x,y,w,h) in faces:
        # x=x-10
        # y=y-10
        cropped_face = img[y-10:y+h+50, x-10:x+w+50]
    return cropped_face

# initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# collecting 200 samples of face from webcam
while True:
    ret, frame = cap.read()
    temp = face_extractor(frame)
    if temp is not None:
        count += 1
        face = cv2.resize(temp, (400, 400))
        # saving file in specified directory with unique name
        file_name_path = '/home/sadabrata/transfer_learning_opencv/dataset/test/sadabrata/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)
        # displaying live count on images
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
    else:
        print("Face not found")
        pass
    if cv2.waitKey(1) == 13 or count == 200: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")