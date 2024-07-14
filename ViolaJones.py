import os
import cv2

#Find the path to xml files that contains trained Haar Cascade models 
cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheyes = os.path.dirname(cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"

#Load our classifier 
faceCascade = cv2.CascadeClassifier(cascPathface)
eyeCascade = cv2.CascadeClassifier(cascPatheyes)

#Open the webcam
video_capture = cv2.VideoCapture(0)

#Get the frames from the webcam stream until we want to close it
#This returns the actual video frame read and a return code (which tells us if we have run out of frames)
while True:
    #Capture frame-by-frame
    ret,frame=video_capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detectMultiScale receives a frame as an argument and runs the classifier cascade over an image.
    #algorithm looks at subregions of image in multiple scales to detect faces of varying sizes
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.05, #how much image size is reduced at each image scale (such as resize larger face to smaller one) (1.05 means 5% size reduction)
                                         minNeighbors=5, #how many neighbors each candidate rectangle should have to retain it. Affects quality of detected faces (high value means fewer detections but high quality)
                                         minSize=(60, 60), #minimum possible object size
                                         flags=cv2.CASCADE_SCALE_IMAGE) #mode of operation

#faces now contains all detections for target image, saved as pixel coordinates
#Each detection defined by top left corner coordinates and width/height of face rectangle
    for (x,y,w,h) in faces:
        #rectangle draws a box over the face
        cv2.rectangle(frame, (x, y), (x + w, y + h),(255,0,0), 2)
        
        #detects eyes and encircles them 
        faceROI = frame[y:y+h,x:x+w]
        eyes = eyeCascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv2.circle(frame, eye_center, radius, (0, 0, 255), 4)
            
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    #Exit script by pressing q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#clena up and release the picture
video_capture.release()
cv2.destroyAllWindows()

    