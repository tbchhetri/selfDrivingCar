import cv2 #image manipulator
import numpy as np
#import matplotlib.pyplot as plt #library courtasy of anaconda

#steps
#1. convert image to grayscale
#2. reduce noise
#3. use gradient
def makeCoordinates (image, lineParameters):
    slope, intercept = lineParameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array ([x1, y1, x2, y2])


def averageSlopeIntercept(image, lines):
    leftFit = []
    rightFit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)#reshaping our 2D arrays
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters [0]
        intercept = parameters [1]
        if slope < 0:
            leftFit.append((slope, intercept))#append write together
        else:
            rightFit.append((slope, intercept))
    leftFitAverage = np.average(leftFit, axis=0)
    rightFitAverage = np.average(rightFit, axis=0)
    leftLine = makeCoordinates(image, leftFitAverage)
    rightLine = makeCoordinates(image, rightFitAverage)
    return np.array([leftLine, rightLine])


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)#converts color of the image to grey as in grey we won't have to worry about RGB
    blur = cv2.GaussianBlur(gray,(5,5), 0) #to reduce noise
    canny = cv2.Canny(blur, 50, 150)#this finds the derivative of the pixel with it's adj pixels to determine wheater there is a sharp change in the pixel intensity. Any derivative below 50 is black and above 150 is white, it is recommended to use these values in 1:3
    return canny

def display_lines (images, lines):
    rightLane = []
    leftLane = []
    middle = []
    line_image = np.zeros_like(images)#makes same dimention as of our image and make them all black
    if lines is not None:
        for x1, y1, x2, y2 in lines:#note line is just declared here
            #x1, y1, x2, y2 = lines.reshape(4)#making 1 dimentional array
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)#draw the line: 1. the image in which you want to draw, 2. first point of the line segment, 3. second point of the line segment, 4. the color of the lines(Blue color here), line thickness
            #print("The coordinates---->", x1, y1, x2, y2)

            if x1 > 600 :
                rightLane.append(x1)
            else:
                leftLane.append(x1)

            #print ("the right lane ----> " , rightLane)
            #print ("The left lane  ----> ", leftLane)
            #print ("The left lane ----> ", rightLane, len(rightLane))
    #print("this is my last resort ", rightLane)
    return line_image, rightLane, leftLane

def regionOfInterest (image):
    height = image.shape[0]
    polygons = np.array([
    [(200, height), (1100, height), (500,250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

#image = cv2.imread('test_image.jpg') #this reads the image and returns it as an array with each pixel's values
#lane_image = np.copy(image) #making a copy of the image so that any changes made will not effect the original image
#canny_image = canny(lane_image)
#cropped_image = regionOfInterest(canny_image)
#lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array ([]), minLineLength=40, maxLineGap=5) #1. image where you want to detect lines, 2 and 3 resolution of hough arrays (so rho and theta values, 4. minimum number of intersections, 5. just a place holder, 6 n 7 length of the line we will accept
#averaged_lines = averageSlopeIntercept (lane_image, lines)
#line_image = display_lines(lane_image, averaged_lines)
#combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)#decrease the pixel intensity of lane_image, keep same intensity for line_image, the last value is not significant: this number gets added to our weighted sum. addWeighted just does a bitwise or of these two images
#cv2.imshow("result", combo_image) #first arg name of the window in which image will be shown, second arg the image you are trying to show
#plt.imshow(cropped_image) #same thing as above
#cv2.waitKey(0)
#plt.show() # time (ms) for which image is to be displayed, here infinitely

cap = cv2.VideoCapture("test2.mp4")
while (cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = regionOfInterest(canny_image)
    #this below return 4 coordinates
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array ([]), minLineLength=40, maxLineGap=5) #1. image where you want to detect lines, 2 and 3 resolution of hough arrays (so rho and theta values, 4. minimum number of intersections, 5. just a place holder, 6 n 7 length of the line we will accept
    averaged_lines = averageSlopeIntercept (frame, lines)
    line_image, rightLane, leftLane = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)#decrease the pixel intensity of lane_image, keep same intensity for line_image, the last value is not significant: this number gets added to our weighted sum. addWeighted just does a bitwise or of these two images
    cv2.imshow("result", combo_image) #first arg name of the window in which image will be shown, second arg the image you are trying to show
    #plt.imshow(cropped_image) #same thing as above
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #here comes the PID

    kp = 1
    ki = 0.23
    kd = 3

    pid_i = 0
    pid_p = 0
    pid_d = 0
    steering = 0

    idealRight = [995]
    idealLeft = [300]
    error = (rightLane[0] - idealRight) - (idealLeft - leftLane [0]) #might have to div by 2
    #print("this is the error-->", error)

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
    print("this is steering-->", steering)
    print("====================================")

cap.release()
cv2.destroyAllWindows()
    #plt.show() # time (ms) for which image is to be displayed, here infinitely
