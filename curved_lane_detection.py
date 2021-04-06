import numpy as np      ## working with arrays
import cv2              ## Image processing - OpenCV

image = cv2.imread('curved_lane.jpg')

## working resolution       ================================================================================
h = 480
w = 640
sz = (w,h)


## canny threshold      ==================================================================================
thresh1 = 30
thresh2 = 100

## roi points       ========================================================================================

# bottom left point
A1 = 0      #x-axis
A2 = 350    #y-axis

# bottom right point
B1 = 640    #x-axis
B2 = 350    #y-axis

# top right point
C1 = 430    #x-axis
C2 = 150   #y-axis

# top left point
D1 = 210    #x-axis
D2 = 150    #y-axis


## set roi from roi points
def roi(image):
    region = np.array([[(A1,A2), (B1, B2), (C1, C2), (D1, D2)]])
    black = np.zeros_like(image)
    cv2.fillPoly(black, region,255)
    return cv2.bitwise_and(image,black)

## line_coordinates
def line_coordinates(image,parameters):


    # try catch block to make sure it runs even if the the arguement given is invalid
    try:
        m,b = parameters
    except TypeError:
        m,b = 0.01,0
    
    y1 = image.shape[0]     #where line starts
    y2 = int(y1*(3/4))      #where line ends
    x1 = int((y1-b)/m)      #line along the lane line
    x2 = int((y2-b)/m)      #line along the lane line

    return np.array([x1,y1,x2,y2])  






# #resize the image resolution into working resolution
# img_rsz = cv2.resize(image,sz,interpolation = cv2.INTER_AREA)

# using video feed
feed = cv2.VideoCapture("cap21.mp4")
while(feed.isOpened()):
    _, frame = feed.read()

    # setting frame resolution as default
    img_rsz = cv2.resize(frame, sz, interpolation = cv2.INTER_AREA)
    
    #duplicate
    img_rsz1 = img_rsz

    ##grayscale image
    grayscale = cv2.cvtColor(img_rsz,cv2.COLOR_RGB2GRAY) 

    ## warp points
    p1 = np.float32([[D1,D2],[C1,C2],[A1,A2],[B1,B2]])  

    ## output warp image to resolution 
    p2 = np.float32([[0,0],[w,0],[0,h],[w,h]])          

    ##image warping for Bird's-Eye View
    warp_points = cv2.getPerspectiveTransform(p1,p2)
    output = cv2.warpPerspective(grayscale, warp_points,(w,h))


    # canny edge detection
    img_canny = cv2.Canny(cv2.GaussianBlur(grayscale,(5,5),2,1),thresh1,thresh2)

    ## roi image
    img_roi = roi(img_canny)
    
    #displaying circles for the roi points
    for x in range (0,4):
        cv2.circle(img_rsz1,(p1[x][0],p1[x][1]),5,(0,0,255),cv2.FILLED)

    # ##different approach for curved lanes lines  
    # ##  split image height
    img1_h = int(h/3)
    img2_h = int(img1_h + h/3)
    img3_h = int(img2_h + h/3)             

    # ##  split original image to three images
    img1 = img_roi[0:img1_h][0:w]
    img2 = img_roi[img1_h:img2_h][0:w]
    img3 = img_roi[img2_h:img3_h][0:w]      #used for determining lane center

    # # #  perform canny edge detection        **canny must be done before roi**
    # # img1_canny = cv2.Canny(img1,thresh1,thresh2)
    # # img2_canny = cv2.Canny(img2,thresh1,thresh2)
    # # img3_canny = cv2.Canny(img3,thresh1,thresh2)

    # lane centering
    blank_img = np.zeros_like(img_rsz1)
    blank_img = cv2.cvtColor(blank_img,cv2.COLOR_BGR2GRAY)
    blank_img[img2_h:img3_h][0:w] = img3 

    ## next step: turn lanes into line segment

    #using Probabilistic Hough transform for identifying the connecting lines
    lines = cv2.HoughLinesP(blank_img,180,np.pi/180,100,np.array([]), minLineLength=50, maxLineGap=10)
    
    #empty arrays for averaging
    right_line = []
    left_line = []

    # try:
    #     x1,y1,x2,y2 = lines[0]
    # except TypeError:
    #     x1,y1,x2,y2 = 0,150,640,150    

    for x1,y1,x2,y2 in lines:
        print(x1,y1)    #1st end point
        print(x2,y2)    #2nd end point

        # display lines in image
        cv2.line(img_rsz1, (x1,y1), (x2,y2), (0,255,0), thickness = 10)
        
        #creating st. line of 1st degree y = mx + b 
        new_lines = np.polyfit((x1,x2),(y1,y2),1)
        
        #slope
        m =  new_lines[0]
        #intercept
        b = new_lines[1]

        if m < 0:
            left_line.append((m,b))
        else:
            right_line.append((m,b))

    #averaging out lines    
    left_line_average = np.average(left_line)
    right_line_average = np.average(right_line)

    left_line = line_coordinates(img_rsz1,left_line_average)
    right_line = line_coordinates(img_rsz1,right_line_average)

    line_array = np.array([left_line,right_line])

    for x1,y2,x2,y2 in line_array:
        cv2.line(img_rsz1,(x1,y1),(x2,y2),(255,255,5),5)


    
    


    # car center line. assumed the camera is placed in the middle of the windshield/car
    center_line = cv2.line (img_rsz1,(320,300),(320,350),(0,0,255),4)


    ## Display Image
    cv2.imshow('result',roi(grayscale))
    cv2.imshow('result1', img1)
    cv2.imshow('result2', img2)
    cv2.imshow('result3', img3)
    cv2.imshow('result4',img_canny)
    cv2.imshow('result5',output)
    cv2.imshow('result6',img_rsz1)
    cv2.waitKey(50)