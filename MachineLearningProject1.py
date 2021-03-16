#implementing the vanilla gradientDescent #the next one implemeting in the next file is the Stochastic gradientDescent
import numpy as np
import matplotlib.pyplot as plt

#essentials
def drawLine(x1, x2):
    ln = plt.plot(x1,x2, color='g')
    plt.pause(0.001)
    ln[0].remove() #this is how you access previous state of a variable

def sigmoid(sig):
    return 1/(1+np.exp(-sig))

def errorCalculator(line_parameter, points, y):
    m = points.shape[0]
    p = sigmoid(points*line_parameter) #p=probability
    #linearCombination = all_points*line_parameter
    #proabilities = sigmoid(linearCombination)
    crossEntropy = -(1/m)*(np.log(p).T*y + np.log(1-p).T*(1-y)) #this equation calculates the error
    return crossEntropy

def gradientDescent(line_parameter, points, y, alpha):
    m = points.shape[0]
    for i in range (500):
        p = sigmoid(points*line_parameter) #the probability
        gradient = points.T*(p-y)*(alpha/m)#alpha = learning rate
        line_parameter = line_parameter - gradient

        w1 = line_parameter.item(0)
        w2 = line_parameter.item(1)
        bias = line_parameter.item(2)

        x1 = np.array([points[:,0].min(), points[:,0].max() ]) #probably only outputs one min and one max
        x2 =  -bias/w2+x1*(-w1/w2) #from the equation of a straight line
        drawLine(x1,x2)


#the top red portion of the graph
n_pts = 100

bias = np.ones(n_pts)

#getting the points
np.random.seed(0)#makes sure we get same random numbers each time we run
random_x1_values = np.random.normal(10,2, n_pts)#1. mean, 2. standard deviation, 3. how many points we want.
random_x2_values = np.random.normal(12,2,n_pts)
top_region = np.array([random_x1_values,random_x2_values, bias]).T #lke hly fuk just by adding ".T"we took the transpose of this array because x1 and x2 were not together like points, I MEAN WOW

#the bottom blue portion of the graph
bottom_region = np.array([np.random.normal(5,2,n_pts), np.random.normal(6,2,n_pts), bias]).T

#YAY the machine learing stuffs
all_points = np.vstack((top_region, bottom_region))#so in the list topRegion Points comes first and then bottomRegion Points

line_parameter = np.matrix(np.zeros(3)).T #matrix is just 2D array, if is capable of stuffs like matrix multiplications. We are taking the transpose to make the no of row of 1st matix equal to no of row of 2nd

#all the red points are classified as 0 and all blues as 1
y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2,1)

#plotting the graphs
_, ax = plt.subplots(figsize = (5,5))#just the size of the figure
ax.scatter(top_region[:,0], top_region[:,1], color='r')
ax.scatter(bottom_region[:,0], bottom_region[:,1], color='b')

gradientDescent(line_parameter, all_points, y, 0.6)

#displaying
plt.show()
print(errorCalculator(line_parameter, all_points, y))


#code form the article

import cv2
import sys
from hand_coded_lane_follower import HandCodedLaneFollower

def save_image_and_steering_angle(video_file):
    lane_follower = HandCodedLaneFollower()
    cap = cv2.VideoCapture(video_file + '.avi')

    try:
        i = 0
        while cap.isOpened():
            _, frame = cap.read()
            lane_follower.follow_lane(frame)
            cv2.imwrite("%s_%03d_%03d.png" % (video_file, i, lane_follower.curr_steering_angle), frame)
            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    save_image_and_steering_angle(sys.argv[1])
