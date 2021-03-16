import numpy as np
import cv2
#import statistics
#import time


def make_coordinates(image,line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        slope,intercept = 0.001,0

    y1 = image.shape[0]
    y2 = int(y1*(1/2))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1, x2, y2])

def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit, axis= 0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image,right_fit_average)
    return np.array([left_line, right_line])



def canny(image):
    # gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    # blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(image, 0, 255)
    return canny

def display_lines(image,lines):
    line_image = np.zeros_like(image)

    middle = []
    #line_image1 = np.zeros_like(image)
    if lines is not None:

        for x1, y1, x2, y2 in lines:

            cv2.line(line_image, (x1, y1), (x2, y2), (255,255, 5), 10)
            print("The coordinates---->", x1, y1, x2, y2)

    return line_image, rightLane, leftLane

cap = cv2.VideoCapture("cap17.mp4")
while(cap.isOpened()):
    rightLane = []
    leftLane = []
    _, frame = cap.read()
    kp = 1
    ki = 0.62
    kd = 3

    pid_i = 0
    pid_p = 0
    pid_d = 0
    steering = 0
    w,h = 640,480
    pts1 = np.float32([[230,200],[390,200],[50,300],[625,300]])
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    output = cv2.warpPerspective(frame,matrix,(w,h))


    for x in range (0,4):
        cv2.circle(frame,(pts1[x][0],pts1[x][1]),5,(0,0,255),cv2.FILLED)

    hsv = cv2.cvtColor(output,cv2.COLOR_BGR2HSV)

    low_yellow = np.array([20,100,100])
    high_yellow = np.array([50,255,200])

    mask = cv2.inRange(hsv,low_yellow,high_yellow)
    yellow = cv2. bitwise_and(frame,frame,mask)

    low_white = np.array([0, 5, 150])
    high_white = np.array([170, 255, 200])

    mask_white = cv2.inRange(hsv,low_white,high_white)
    white = cv2. bitwise_and(frame,frame,mask_white)
    cv2.imshow('result11',mask_white)

    canny_image = canny(mask_white)
    ptss1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    ptss2 = np.float32([[230,200],[390,200],[50,300],[625,300]])
    matrixx = cv2.getPerspectiveTransform(ptss1,ptss2)
    output1 = cv2.warpPerspective(canny_image,matrixx,(w,h))
    # cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(output1, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image, rightLane, leftLane = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 1, line_image, 0.7, 1)
    x11,y11 = average_slope_intercept(frame,lines)
    #  print (x11,y11)
    _, _, leftLane, _ = lines[0][0]
    _, _, rightLane, _ = lines[1][0]
    botmidx = int((x11[0]+y11[0])/2)
    botmidy = int((x11[1]+y11[1])/2)
    topmidx = int((x11[2]+y11[2])/2)
    topmidy = int((x11[3]+y11[3])/2)
    mid_line = cv2.line(combo_image, (botmidx,botmidy), (topmidx, topmidy+100), (255,255, 5), 1,)

    center_line = cv2.line (combo_image,(320,300),(320,480),(0,0,255),4)

    cv2.imshow("combo_image",combo_image)


    cv2.imshow("frame",frame)
    cv2.imshow("output1",output1)
    #print("this is the shit -------->", pts1[1][0])

    rightLane = np.asarray([rightLane])
    leftLane = np.asarray([leftLane])

    print ("The left lane ----> ", leftLane)
    print ("The right lane ----> ", rightLane)
    #middle = (rightLane[0]-leftLane[0])/2
    #print ("The middle is -----> ", statistics.mean(leftLane[0]))
    #print ("The left new lane ----> ", leftLane)
    #print ("The right lane ----> ", rightLane)


    #carMid = rightLane - leftLane
    idealRight = [600]
    idealLeft = [75]

    #error between desired and real position
    error = (rightLane[0] - idealRight) - (idealLeft - leftLane [0]) #might have to div by 2
    #print("this is error-->", error)
    #Proportional
    pid_p = kp*error

    #integral
    if(-3 < error < 3):
        pid_i += ki*error

    #derivative
    #pid_d = kd*((error-previous_error)/elapsedTime)
    pid_d = 0
    PID = pid_p + pid_i + pid_d

    steering += PID
    #print("The steering angle is ---->", steering
    #loop (pts1)
    #print("this is steering-->", steering)
    print("====================================")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
