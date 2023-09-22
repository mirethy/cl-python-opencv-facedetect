import cv2

#load pre trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose image to detect face in
#img = cv2.imread('52-05.jpg')
#img = cv2.imread('img2p.jpg')
webcam = cv2.VideoCapture(1)  #detect face in video
#key = cv2.waitKey(1)

#iterate over frames
while True:
    successful_frame_read, frame = webcam.read() #get current frame

#make it grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.waitKey(1)

#detect faces
    face_coordinates = trained_face_data.detectMultiScale(gray_img)
    # print(face_coordinates)

#draw rectangle around face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 3)

#show image
    cv2.imshow('face detector', frame)
    key = cv2.waitKey(1)
    
    #stop if q is pressed
    if key==81 or key ==113:
        break

#release the videocapture object
webcam.release()

print('code completed')