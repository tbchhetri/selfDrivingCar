#uses the model developed from the machine learning algorithm to control the steering of
#the car by send pwm signal to the steering motor of the car
import numpy as np
import cv2
import tensorflow as tf
#from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#model = keras.models.load_model('/Volumes/MyData/Data/College/Winter20/behavioralCloning/model.h5')
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)

#this is the model output from the ml algorithm
model = tf.keras.models.load_model("model.h5")
#print(model.summary())

#image = cv2.imread('road3.jpg') #this reads the image and returns it as an array with each pixel's values

#this function preprocess all the incoming images and makes them like the image we trained the ML algo with
def img_preprocess(img):
  #img = mpimg.imread(img)
  img = img[60:135,:,:] #we are cropping the height of the image since other than 60-135 we dont need those infos
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) #Y-luminosity, and UV. This is recommended by NVIDIA while running their Neural
  img = cv2.GaussianBlur(img, (3,3), 0) #(3,3) is the kernal size for the blur, this is done to reduce the noise in image
  img = cv2.resize(img, (200,66)) #this size is also recommended by NVIDIA
  img = img/255
  return img.reshape( 66,200, 3)

#plt.imshow(img)
#cv2_imshow(img) #apparantly google doesn't support cv2.imshow, so this is the alternative

#for left motor
enA = 13
in1 = 2
in2 = 3
#for right one
#enB = 12
#in3 = 17
#in4 = 27

error = 100 #the proportional controller

#required gpio setup for the RPi
GPIO.setup(enA, GPIO.OUT)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)

pwmA = GPIO.PWM(enA, 100)
pwmA.start(0)

#GPIO.setup(enB, GPIO.OUT)
#GPIO.setup(in3, GPIO.OUT)
#GPIO.setup(in4, GPIO.OUT)

#pwmB = GPIO.PWM(enA, 100)
#pwmB.start(0)

cap = cv2.VideoCapture("ohiotest.mp4")
while (cap.isOpened()):
    _, frame = cap.read() #extracts every frame from the image and saves it in frame

    img = img_preprocess(frame)

    img = np.asarray([img])
    prediction1 = model.predict([img])
    print("The steering angle is: ", str(prediction1))

    if prediction1 > 0:
        #if positive move right
        GPIO.output(in3, 0)
        GPIO.output(in4, 1)
        pwmB.ChangeDutyCycle(0)

        GPIO.output(in1, 0)
        GPIO.output(in2, 1)
        pwmA.ChangeDutyCycle(prediction1*error)

    else:
        GPIO.output(in3, 0)
        GPIO.output(in4, 1)
        pwmB.ChangeDutyCycle(prediction1*error)

        GPIO.output(in1, 0)
        GPIO.output(in2, 1)
        pwmA.ChangeDutyCycle(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
