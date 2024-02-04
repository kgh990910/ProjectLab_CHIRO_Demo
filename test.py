#----------------------------------------------------------------------
# CAU CHIRO : KGH
# Raspberry pi 3B, bullseye 64bit
# python3 == python 3.9
# if you were to participate in CAU MJY X CHIRO projectLab
# the use of this code is prohibited
#----------------------------------------------------------------------

import RPi.GPIO as GPIO # for use gpio
import cv2 # opencv ver : 4.6.0
import numpy as np 
import time # for pid control
import math # for use pi
from skimage.metrics import structural_similarity as ssim # for compare image
import constant # for use constant var, check constant.py
import threading # for multi_thread

#-----------------------------------------------------------------------
#-------------- for detect traffic light, human and sign ---------------
#-----------------------------------------------------------------------

school_1 = cv2.imread('./school_1.png')
school_2 = cv2.imread('./school_2.png')
tunnel = cv2.imread('./tunnel.png')
traffic_left = cv2.imread('./left.png')
traffic_straight = cv2.imread('./straight.png')
traffic_right = cv2.imread('./right.png')
traffic_stop = cv2.imread('./stop.png')
school_1 = cv2.cvtColor(school_1, cv2.COLOR_BGR2GRAY)
school_2 = cv2.cvtColor(school_2, cv2.COLOR_BGR2GRAY)
tunnel = cv2.cvtColor(tunnel, cv2.COLOR_BGR2GRAY)
traffic_left = cv2.cvtColor(traffic_left, cv2.COLOR_BGR2GRAY)
traffic_straight = cv2.cvtColor(traffic_straight, cv2.COLOR_BGR2GRAY)
traffic_right = cv2.cvtColor(traffic_right, cv2.COLOR_BGR2GRAY)
traffic_stop = cv2.cvtColor(traffic_stop, cv2.COLOR_BGR2GRAY)
human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

#-----------------------------------------------------------------------
#--------------------- set GPIO pin Num --------------------------------
#-----------------------------------------------------------------------


constant.ENA = 12
constant.IN1 = 5
constant.IN2 = 6
# constant.IN3 = 22
constant.IN3 = 17
constant.IN4 = 27
constant.ENB = 13
constant.LED = 25

#-----------------------------------------------------------------------
#--------------------- set constant var --------------------------------
#-----------------------------------------------------------------------

constant.HIGH = 1
constant.LOW = 0
constant.FORWARD = 1
constant.STOP = 0
constant.BACKWARD = -1
constant.L = 0
constant.R = 1

constant.FRAME_VER=144
constant.FRAME_HOR=256
constant.ROI_VER=int(constant.FRAME_VER/2)
constant.ROI_HOR=int(constant.FRAME_HOR/3)

constant.LEFT_WEIGHT_TARGET = 17
constant.RIGHT_WEIGHT_TARGET = 42
constant.TRACK_WEIGHT_TARGET = 127
constant.WEIGHT_OFFSET = 10
constant.BLACK= (255, 255, 255)

#-----------------------------------------------------------------------
#--------------------- set glo variable --------------------------------
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
#--------------------- set pid variable --------------------------------
#-----------------------------------------------------------------------

LEFT_SPEED = 30
RIGHT_SPEED = 30
# LEFT_SPEED = 0
# RIGHT_SPEED = 0

state = 0 # 0: stop, 1: left, 2: right, 3: straight

# pwmL, pwmR = 0, 0

#-----------------------------------------------------------------------
#--------------------- gpio init function ------------------------------
#-----------------------------------------------------------------------

def gpioInit():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(constant.ENA, GPIO.OUT)
    GPIO.setup(constant.IN1, GPIO.OUT)
    GPIO.setup(constant.IN2, GPIO.OUT)
    GPIO.setup(constant.IN3, GPIO.OUT)
    GPIO.setup(constant.IN4, GPIO.OUT)
    GPIO.setup(constant.ENB, GPIO.OUT)
    GPIO.setup(constant.LED, GPIO.OUT)

    pwmL=GPIO.PWM(constant.ENA, 1000)
    pwmR=GPIO.PWM(constant.ENB, 1000)

    pwmL.start(0)
    pwmR.start(0)

    return pwmL, pwmR

#-----------------------------------------------------------------------
#--------------------- camera init function ----------------------------
#-----------------------------------------------------------------------

def camerInit():
    capture=cv2.VideoCapture(0)
    print("Pi CAM Operate")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    capture.set(cv2.CAP_PROP_FPS,60)
    print("Pi CAM Setting Finish")

    return capture

#-----------------------------------------------------------------------
#--------------------- control motor function --------------------------
#-----------------------------------------------------------------------

def setMotor(CH, pwm, SPEED, DIRECTION):
    pwm.ChangeDutyCycle(SPEED)
    if CH==constant.R:
        if DIRECTION == constant.STOP:
            GPIO.output(constant.IN1, constant.LOW)
            GPIO.output(constant.IN2, constant.LOW)
        elif DIRECTION == constant.FORWARD:
            GPIO.output(constant.IN1, constant.HIGH)
            GPIO.output(constant.IN2, constant.LOW)
        elif DIRECTION == constant.BACKWARD:
            GPIO.output(constant.IN1, constant.LOW)
            GPIO.output(constant.IN2, constant.HIGH)
    elif CH==constant.L:
        if DIRECTION == constant.STOP:
            GPIO.output(constant.IN3, constant.LOW)
            GPIO.output(constant.IN4, constant.LOW)
        elif DIRECTION == constant.BACKWARD:
            GPIO.output(constant.IN3, constant.HIGH)
            GPIO.output(constant.IN4, constant.LOW)
        elif DIRECTION == constant.FORWARD:
            GPIO.output(constant.IN3, constant.LOW)
            GPIO.output(constant.IN4, constant.HIGH)
   
#-----------------------------------------------------------------------
#--------------------- control led function ----------------------------
#-----------------------------------------------------------------------
     
def ledOnOff(STATE):
    GPIO.output(constant.LED, STATE)
    if STATE == constant.HIGH:
        print("LED ON")
    elif STATE == constant.LOW:
        print("LED OFF")

#-----------------------------------------------------------------------
#--------------------- detect school zone ------------------------------
#-----------------------------------------------------------------------

def checkSchoolzone(img, pts):
    (x, y, w, h) =cv2.boundingRect(pts)
    pt1 = (x,y)
    pt2 = (x + w, y + h)
    rect_area=w*h
    if rect_area >= 1500 and rect_area <= 5000: # need change size
        roi=img[y:y+h, x:x+w]
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
        roi_grey=cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ssim_index_1, _ = ssim(roi_grey, cv2.resize(school_1,(w, h)), full=True)
        ssim_index_2, _ = ssim(roi_grey, cv2.resize(school_2,(w, h)), full=True)
        if ssim_index_1 > 0.2:
            cv2.putText(img, "start", (pt1[0], pt1[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
        if ssim_index_2 > 0.2:
            cv2.putText(img, "end", (pt1[0], pt1[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

#-----------------------------------------------------------------------
#--------------------- detect tunnel sign ------------------------------
#-----------------------------------------------------------------------

def checkTunnel(img, pts):
    (x, y, w, h) =cv2.boundingRect(pts)
    pt1 = (x,y)
    pt2 = (x + w, y + h)
    rect_area=w*h
    if rect_area >= 500 and rect_area <= 1000: #need change size
        roi=img[y:y+h, x:x+w]
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
        roi_grey=cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ssim_index, _ = ssim(roi_grey, cv2.resize(tunnel,(w, h)), full=True)
        if ssim_index > 0.5:
            cv2.putText(img, "tunnel", (pt1[0], pt1[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

#-----------------------------------------------------------------------
#--------------------- detect human 000000000 --------------------------
#-----------------------------------------------------------------------    

def detectHuman(frame):
    humans = human_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # cv2.imshow('Human Detection', frame)

#-----------------------------------------------------------------------
#------------------ detect traffic light "stop" ------------------------
#-----------------------------------------------------------------------

def checkStop(img, pts):
    (x, y, w, h) =cv2.boundingRect(pts)
    pt1 = (x,y)
    pt2 = (x + w, y + h)
    rect_area=w*h
    if rect_area >= 500 and rect_area <= 1000: #need change size
        roi=img[y:y+h, x:x+w]
        r=g=0
        for p in range(h):
            for q in range(w):
                r=roi[p,q,0]+r
                g=roi[p,q,1]+g
        if r > g:
            cv2.putText(img, "stop", (pt1[0], pt1[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

#-----------------------------------------------------------------------
#------------------ detect traffic light "dir" -------------------------
#-----------------------------------------------------------------------

def trafficLight(img, pts):
    (x, y, w, h) =cv2.boundingRect(pts)
    pt1 = (x,y)
    pt2 = (x + w, y + h)
    rect_area=w*h
    if rect_area >= 300 and rect_area <= 1000:
        roi=img[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (5,5), sigmaX=1.0)
        roi_grey =cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ssim_index_s, _ = ssim(roi_grey, cv2.resize(traffic_straight,(w, h)), full=True)
        ssim_index_r, _ = ssim(roi_grey, cv2.resize(traffic_right,(w, h)), full=True)
        ssim_index_l, _ = ssim(roi_grey, cv2.resize(traffic_left,(w, h)), full=True)
        if ssim_index_s > 0.4:
            cv2.putText(img, "straight", (pt1[0], pt1[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
        if ssim_index_r > 0.4:
            roi=cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_left = roi[0:h, 0:int(w/2)]
            roi_right = roi[0:h, int(w/2):w]
            h1,w1, _ = roi_left.shape
            h2,w2, _ = roi_right.shape
            g1=g2=r1=r2=0

            for p in range(h1):
                for q in range(w1):
                    g1=roi_left[p,q,1]+g1
                    r1=roi_left[p,q,0]+r1
            for p in range(h2):
                for q in range(w2):
                    g2=roi_right[p,q,1]+g2
                    r2=roi_right[p,q,0]+r2
            if r1 < r2:
                cv2.putText(img, "left", (pt1[0], pt1[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
            elif r1 > r2:
                cv2.putText(img, "right", (pt1[0], pt1[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
        if ssim_index_l > 0.4:
            roi=cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_left = roi[0:h, 0:int(w/2)]
            roi_right = roi[0:h, int(w/2):w]
            h1,w1, _ = roi_left.shape
            h2,w2, _ = roi_right.shape
            g1=g2=r1=r2=0

            for p in range(h1):
                for q in range(w1):
                    g1=roi_left[p,q,1]+g1
                    r1=roi_left[p,q,0]+r1
            for p in range(h2):
                for q in range(w2):
                    g2=roi_right[p,q,1]+g2
                    r2=roi_right[p,q,0]+r2
            if r1 < r2:
                cv2.putText(img, "left", (pt1[0], pt1[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
            elif r1 > r2:
                cv2.putText(img, "right", (pt1[0], pt1[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

#-----------------------------------------------------------------------
#------------------ shape detect function ----- ------------------------
#-----------------------------------------------------------------------

def detect(frame_blur, frame_color):
    detectHuman(frame_blur)
    canny=cv2.Canny(frame_blur,50,100)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.imshow("canny", canny)

    for cnt in contours:
        approx =cv2.approxPolyDP(cnt, cv2.arcLength(cnt, True)*0.04, True)
        vtc = len(approx)

        if vtc == 3:
            checkTunnel(frame_color, cnt)
        elif vtc == 4:
            checkSchoolzone(frame_color, cnt)
        elif vtc == 7:
            trafficLight(frame_color, cnt)
        else:
            area = cv2.contourArea(cnt)
            _, radius = cv2.minEnclosingCircle(cnt)

            if area != 0:
                ratio = radius * radius * math.pi / area

                if int(ratio) == 1:
                    checkStop(frame_color, cnt)
    # cv2.imshow('test', frame_color)


def getWeight(roi_frame, WEIGHT):
    contours, hierarchy = cv2.findContours(roi_frame, 1, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            WEIGHT = cx
            cv2.line(roi_frame,(cx,0),(cx,constant.ROI_VER),(255,0,0),1) 

    return WEIGHT

def line_dict():
    global prev_time, ret, state

    try:
        capture = camerInit()
        LEFT_WEIGHT = 0
        RIGHT_WEIGHT = 0 

        prev_time = time.time()

        while True:
            ret, frame = capture.read()
        
            # setMotor(constant.L, pwmL, LEFT_SPEED, constant.STOP)
            # setMotor(constant.R, pwmR, RIGHT_SPEED, constant.STOP)

            resized_frame = cv2.resize(frame, dsize=(constant.FRAME_HOR, constant.FRAME_VER), interpolation = cv2.INTER_AREA)
            resized_frame = cv2.flip(resized_frame, -1)
            frame_gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            frame_blur = cv2.GaussianBlur(frame_gray, (5,5), sigmaX=1.0)

            # detect(frame_blfur, resized_frame)

            ret, frame_binary = cv2.threshold(frame_blur, 150, 255, cv2.THRESH_BINARY)

            resized_frame = cv2.resize(frame, dsize=(constant.FRAME_HOR, constant.FRAME_VER), interpolation = cv2.INTER_AREA)
            resized_frame = cv2.flip(resized_frame, -1)
            frame_gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            frame_blur = cv2.GaussianBlur(frame_gray, (5,5), sigmaX=1.0)

            # detect(frame_blfur, resized_frame)

            left_roi = frame_binary[constant.FRAME_VER - int(constant.ROI_VER) : constant.FRAME_VER + int(constant.ROI_VER), 0:constant.ROI_HOR]
            right_roi = frame_binary[constant.FRAME_VER - int(constant.ROI_VER) : constant.FRAME_VER + int(constant.ROI_VER), constant.FRAME_HOR-constant.ROI_HOR:constant.FRAME_HOR]
            track_roi = frame_binary[constant.FRAME_VER - int(constant.ROI_VER) : constant.FRAME_VER + int(constant.ROI_VER), 0:constant.FRAME_HOR]

            LEFT_WEIGHT = getWeight(left_roi, LEFT_WEIGHT)
            RIGHT_WEIGHT = getWeight(right_roi, RIGHT_WEIGHT)

            TRACK_WEIGHT = int((LEFT_WEIGHT+RIGHT_WEIGHT+(constant.FRAME_HOR*2/3))/2)
            cv2.line(track_roi,(constant.TRACK_WEIGHT_TARGET+constant.WEIGHT_OFFSET, 0),(constant.TRACK_WEIGHT_TARGET+constant.WEIGHT_OFFSET, constant.ROI_VER),(255,0,0),1)
            cv2.line(track_roi,(constant.TRACK_WEIGHT_TARGET-constant.WEIGHT_OFFSET, 0),(constant.TRACK_WEIGHT_TARGET-constant.WEIGHT_OFFSET, constant.ROI_VER),(255,0,0),1)
            cv2.line(track_roi,(TRACK_WEIGHT, 0),(TRACK_WEIGHT, constant.ROI_VER),(255,0,0),1)

            if TRACK_WEIGHT < (constant.TRACK_WEIGHT_TARGET - constant.WEIGHT_OFFSET):
                print("LEFT")
                state = 1

            elif TRACK_WEIGHT > (constant.TRACK_WEIGHT_TARGET + constant.WEIGHT_OFFSET):
                print("right")
                state = 2
                
            else:
                print("strait")
                state = 3

            now = time.time()
            print(now - prev_time)
            prev_time = now

            if cv2.waitKey(1)>0:
                break

    except KeyboardInterrupt:
        pass
    ledOnOff(constant.LOW)
    pwmL.stop()
    pwmR.stop()
    #cv2.destoryAllWindows()
    GPIO.cleanup()
    capture.release()

    print("\nProcess END\n")

def motor_control():
    global pwmL, pwmR

    try:
        pwmL, pwmR = gpioInit()

        while True:

            if state == 1: # Left
                setMotor(constant.L, pwmL, int(LEFT_SPEED*2/3), constant.FORWARD) #
                setMotor(constant.R, pwmR, RIGHT_SPEED, constant.FORWARD)

            elif state == 2: # Right
                setMotor(constant.L, pwmL, LEFT_SPEED, constant.FORWARD)
                setMotor(constant.R, pwmR, int(RIGHT_SPEED*2/3), constant.FORWARD) #
            
            else: # Straight
                setMotor(constant.L, pwmL, LEFT_SPEED, constant.FORWARD)
                setMotor(constant.R, pwmR, RIGHT_SPEED, constant.FORWARD)

            if cv2.waitKey(1)>0:
                break

    except KeyboardInterrupt:
        pass
    ledOnOff(constant.LOW)
    pwmL.stop()
    pwmR.stop()
    #cv2.destoryAllWindows()
    GPIO.cleanup()
    capture.release()

    print("\nProcess END\n")

def main_func():

    t1 = threading.Thread(target=line_dict)
    t2 = threading.Thread(target=motor_control)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

#-----------------------------------------------------------------------
#-------------------------- Main Function ------------------------------
#-----------------------------------------------------------------------

if __name__ == "__main__":
    main_func()

# try:
#     pwmL, pwmR = gpioInit()
#     capture = camerInit()
    
    
#     # ledOnOff(constant.HIGH)

#     # setMotor(constant.L, pwmL, LEFT_SPEED, constant.FORWARD)
#     # setMotor(constant.R, pwmR, RIGHT_SPEED, constant.FORWARD)
#     # time.sleep(5)
#     prev_time = time.time()
#     while True:
        
#         ret, frame = capture.read()
        
#         # setMotor(constant.L, pwmL, LEFT_SPEED, constant.STOP)
#         # setMotor(constant.R, pwmR, RIGHT_SPEED, constant.STOP)

        

# #----------------------------------------------------------------------------------------------------
# #---------------------------- line weight detect-----------------------------------------------------
# #----------------------------------------------------------------------------------------------------

#         ret, frame_binary = cv2.threshold(frame_blur, 150, 255, cv2.THRESH_BINARY)

#         # center_roi = frame_binary[constant.FRAME_VER - constant.ROI_VER : constant.FRAME_VER, int(constant.ROI_HOR): constant.FRAME_HOR - int(constant.ROI_HOR)]
#         # center_roi = cv2.Canny(center_roi, 100, 200)
#         # he, wi = center_roi.shape
#         # center_line = cv2.HoughLines(center_roi, rho = 1, theta = np.pi/180.0, threshold = 85)
#         # slope = 0
#         # if center_line is not None:
#         #     cnt = 0 
#         #     for line in center_line:
#         #         rho, theta=line[0]
#         #         c=np.cos(theta)
#         #         s=np.sin(theta)
#         #         x0=c*rho
#         #         y0=s*rho
#         #         x1=int(x0+1000*(-s))
#         #         y1=int(y0+1000*(c))
#         #         x2=int(x0-1000*(-s))
#         #         y2=int(y0-1000*(c))
#         #         cv2.line(center_roi,(x1,y1),(x2,y2),(0,255,0),2)
#         #         if x2-x1!=0:
#         #             slope=slope + (y2-y1)/(x2-x1)
#         #             cnt = cnt + 1
#         #     if cnt != 0:
#         #         slope = slope/cnt
#         #     print("\t\t\t\t"+str(slope))
                

#         # left_roi = frame_binary[constant.FRAME_VER - constant.ROI_VER : constant.FRAME_VER, 0:constant.ROI_HOR]
#         # right_roi = frame_binary[constant.FRAME_VER - constant.ROI_VER : constant.FRAME_VER, constant.FRAME_HOR-constant.ROI_HOR:constant.FRAME_HOR]
#         # track_roi = frame_binary[constant.FRAME_VER - constant.ROI_VER : constant.FRAME_VER, 0:constant.FRAME_HOR]
#         left_roi = frame_binary[constant.FRAME_VER - int(constant.ROI_VER) : constant.FRAME_VER + int(constant.ROI_VER), 0:constant.ROI_HOR]
#         right_roi = frame_binary[constant.FRAME_VER - int(constant.ROI_VER) : constant.FRAME_VER + int(constant.ROI_VER), constant.FRAME_HOR-constant.ROI_HOR:constant.FRAME_HOR]
#         track_roi = frame_binary[constant.FRAME_VER - int(constant.ROI_VER) : constant.FRAME_VER + int(constant.ROI_VER), 0:constant.FRAME_HOR]

#         # left_contours, hierarchy = cv2.findContours(left_roi, 1, cv2.CHAIN_APPROX_NONE)
#         # right_contours, hierarchy = cv2.findContours(right_roi, 1, cv2.CHAIN_APPROX_NONE)

#         LEFT_WEIGHT = getWeight(left_roi, LEFT_WEIGHT)
#         RIGHT_WEIGHT = getWeight(right_roi, RIGHT_WEIGHT)

#         # TRACK_WEIGHT = int((LEFT_WEIGHT+RIGHT_WEIGHT+(constant.FRAME_HOR/2))/2)
#         TRACK_WEIGHT = int((LEFT_WEIGHT+RIGHT_WEIGHT+(constant.FRAME_HOR*2/3))/2)
#         cv2.line(track_roi,(constant.TRACK_WEIGHT_TARGET+constant.WEIGHT_OFFSET, 0),(constant.TRACK_WEIGHT_TARGET+constant.WEIGHT_OFFSET, constant.ROI_VER),(255,0,0),1)
#         cv2.line(track_roi,(constant.TRACK_WEIGHT_TARGET-constant.WEIGHT_OFFSET, 0),(constant.TRACK_WEIGHT_TARGET-constant.WEIGHT_OFFSET, constant.ROI_VER),(255,0,0),1)
#         cv2.line(track_roi,(TRACK_WEIGHT, 0),(TRACK_WEIGHT, constant.ROI_VER),(255,0,0),1)
#         # setMotor(constant.L, pwmL, int(LEFT_SPEED/3), constant.FORWARD) #
#         # setMotor(constant.R, pwmR, RIGHT_SPEED, constant.FORWARD)
#         # setMotor(constant.L, pwmL, LEFT_SPEED, constant.FORWARD)
#         # setMotor(constant.R, pwmR, int(RIGHT_SPEED/3), constant.FORWARD) #

# #----------------------------------------------------------------------------------------------------
# #----------------------------------- Set PID --------------------------------------------------------
# #----------------------------------------------------------------------------------------------------
#         # print(str(LEFT_WEIGHT) + "\t\t" + str(TRACK_WEIGHT) + "\t\t" + str(RIGHT_WEIGHT))
#         if TRACK_WEIGHT < (constant.TRACK_WEIGHT_TARGET - constant.WEIGHT_OFFSET):
#             print("LEFT")
#             # print(time.time())
#             setMotor(constant.L, pwmL, int(LEFT_SPEED*2/3), constant.FORWARD) #
#             setMotor(constant.R, pwmR, RIGHT_SPEED, constant.FORWARD)

#             # time.sleep(0.05)
#             # setMotor(constant.L, pwmL, LEFT_SPEED, constant.FORWARD)
#             # setMotor(constant.R, pwmR, RIGHT_SPEED, constant.FORWARD)
#         elif TRACK_WEIGHT > (constant.TRACK_WEIGHT_TARGET + constant.WEIGHT_OFFSET):
#             print("right")
#             # print(time.time())
#             setMotor(constant.L, pwmL, LEFT_SPEED, constant.FORWARD)
#             setMotor(constant.R, pwmR, int(RIGHT_SPEED*2/3), constant.FORWARD) #

#             # time.sleep(0.05)
#             # setMotor(constant.L, pwmL, LEFT_SPEED, constant.FORWARD)
#             # setMotor(constant.R, pwmR, RIGHT_SPEED, constant.FORWARD)
#         else:
#             print("strait")
#             # print(time.time())
#             setMotor(constant.L, pwmL, LEFT_SPEED, constant.FORWARD)
#             setMotor(constant.R, pwmR, RIGHT_SPEED, constant.FORWARD)
           
# # # only slope        
# #         if slope > 0:
# #             print("slope LEFT")
# #             setMotor(constant.L, pwmL, LEFT_SPEED, constant.BACKWARD)
# #             setMotor(constant.R, pwmR, RIGHT_SPEED, constant.FORWARD)
# #             time.sleep(0.2)
# #         elif slope < 0:
# #             print("slope RIGHT")
# #             setMotor(constant.L, pwmL, LEFT_SPEED, constant.FORWARD)
# #             setMotor(constant.R, pwmR, RIGHT_SPEED, constant.BACKWARD)
# #             time.sleep(0.2)
# #         # setMotor(constant.L, pwmL, LEFT_SPEED, constant.FORWARD)
# #         # setMotor(constant.R, pwmR, RIGHT_SPEED, constant.STOP)
# #         # cv2.imshow("left", left_roi)
# #         # cv2.imshow("right", right_roi)
#         cv2.imshow("track", track_roi)
# #         # time.sleep(0.1)
# #         # setMotor(constant.L, pwmL, LEFT_SPEED, constant.STOP)
# #         # setMotor(constant.R, pwmR, RIGHT_SPEED, constant.STOP)
#         now = time.time()
#         # print(now - prev_time)
#         prev_time = now
#         # cv2.imshow("track", center_roi)
#         # cv2.imwrite("image.png", frame)
#         # cv2.imshow("image", frame_blur)
#         if cv2.waitKey(1)>0:
#             break



# except KeyboardInterrupt:
#     pass
# ledOnOff(constant.LOW)
# pwmL.stop()
# pwmR.stop()
# #cv2.destoryAllWindows()
# GPIO.cleanup()
# capture.release()

# print("\nProcess END\n")