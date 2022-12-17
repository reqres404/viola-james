import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0) #uses camera of laptop

mpHands = mp.solutions.hands#just a formality to use Media pipe
hands = mpHands.Hands(False)#the false statement assures that tracking takes place only when a hand appears and makes the program faster
mpDraw = mp.solutions.drawing_utils#this functions is basically to avoid a lot of maths that goes into configuring 21 landmarks and lines conecting them

#setting previous and current time to manipulate frame speed
pTime =0 #previous time 
cTime = 0 #current time 

while True:
    success ,img = cap.read() # Gives us a frame to work with
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#to convert the iamge into RgB is quite important as the moudle supports RGB images
    results = hands.process(imgRGB)#this helps in processing the frame for us
    #print (results.multi_hand_landmarks)#gives a live co-ordinates of our hands

    if  results.multi_hand_landmarks :#this code helps us to know each hands landmarks spereratly
        for handLMS in results.multi_hand_landmarks:
            for id,lm in enumerate(handLMS.landmark):#it gives a sepcific number to each finger
                #print(id,lm)
                #for conversion of the co-ordinates to pixel form following steps are required
                h,w,c = img.shape#this is gives us aspect of the image
                cx ,cy = int(lm.x*w),int(lm.y*h)#thid would be the requied positon
                print(id,cx,cy)
                #this if blocks helps in just focusing or pointing out one parameter and highlight it.
                if id == 4:
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)

            mpDraw.draw_landmarks(img,handLMS,mpHands.HAND_CONNECTIONS)#this is the function that locates parameters for  us
                                        #HAND_CONNECTIONS is used to connect the lines to the red dots
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    #Displays the fps on the screen
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)

    cv2.imshow("Image",img)#to display the content on screen
    cv2.waitKey(1)

