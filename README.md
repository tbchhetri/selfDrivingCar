# Self-driving Car
![](/images/pic01.jpg)

**Project Summary**

- Trained DNN to identify common traffic signs and trained a Machine Learning architecture based on The Nvidia Model to predict steering angles of a single laned road
- Utilized a lane detection algorthm to detect lanes and drive the car maintaining mid lane steering using PID control
- Equipped Raspberry Pi as the main processing unit and used it to communicate with Pi camera and motors
- Used Lidar to accurately detect nearby objects and take actions accordingly

## Resources used

**[Neural networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)** This video series was used to understand Neural Networks.

**[NVIDIA CNN Model: ](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)** The Machine Learning model for determining the steering angle was based on the parameters suggested by the NVIDIA CNN model.

**[The Complete Self-Driving Car: ](https://www.udemy.com/course/applied-deep-learningtm-the-complete-self-driving-car-course/)** This course was used to learn about Lane detection algorithm and different Machine Learning types, techniques and algorithms. The [NVIDIA_Model.ipynb](/NVIDIA_Model.ipynb) was also inspired from this. 

**[self-driving-car-sim: ](https://github.com/udacity/self-driving-car-sim)** This simulater was used to gather training data for the ML model.

**[Complete Self Driving Car - IOT Project (Final Semester): ](https://youtu.be/Xr0_vScJD8o)** This video was used to get ideas about other people's implemetation of the ML model.

**[OpenCV for Python Developers: ](https://www.linkedin.com/learning/opencv-for-python-developers/get-started-with-opencv-and-python?u=57694233)** This course was used to understand OpenCV and use it in our Lane Detection and ML application

**[DeepPiCar: ](https://towardsdatascience.com/deeppicar-part-1-102e03c83f2c)** This article was used to understant different implemetation of the ML model and new tecniques to Lane Detections.

**[Intro to TinyML Part1: Training a Neural Network for Arduino in TensorFlow Digi-Key Electronics: ](https://www.youtube.com/watch?v=BzzqYNYOcWc)** This video was used to understand the alternatives to using Raspberry Pi

**[Self Driving AI in 100 lines of code Raspberry Pi: ](AI in 100 lines of code | Raspberry Pi)** This video was used to get ideas about other people's implemetation of the ML model and use of Blynk app to control the car!

**[Lane Detection and Tracking with MATLAB: ](https://www.youtube.com/watch?v=SFqAAseL_1g)** This video was used to evaluate the alternatives of using MATLAB for Lane Detection.

**[I2C with Arduino and Raspberry Pi - Two Methods: ](https://www.youtube.com/watch?v=me7mhrRbspk)** This video was used to understand the Master-Slave setup for Raspberry Pi and Arduino UNO.

**[Arduino-Python3 Command API: ](https://pypi.org/project/arduino-python3/)** This article was used to implement python codes directly on Arduino UNO.

**[How to connect your “L298N Dual H-Bridge Motor Controller” to “Arduino Uno”: ](https://www.youtube.com/watch?v=OkHR1BZCcqA)** This video was used to connect the H-Bridge with Arduino UNO

**[FPGA vs GPU for Machine Learning Applications: Which one is better?: ](https://www.aldec.com/en/company/blog/167--fpgas-vs-gpus-for-machine-learning-applications-which-one-is-better)** This article was used to evaluate the alternatives of using FPGA for ML.

**[How to use your trained model - Deep Learning basics with Python, TensorFlow and Keras p.6: ](https://www.youtube.com/watch?v=A4K6D_gx2Iw&t=4s)** This video was used to understand an approach to use a developed ML model.

**[Robotic Car, Closed Loop Control Example: ](https://www.youtube.com/watch?v=LfydfvHyikM)** This video was used to understand the PID contolls on simple car.

**[PID Balance+Ball full explanation & tuning: ](https://www.youtube.com/watch?v=JFTJ2SS4xyA)** This video was used to understand implementation of PID in Ball Balancing System.

**[PID brushless motor control tutorial: ](https://www.youtube.com/watch?v=AN3yxIBAxTA)** This video was used to understand implementation of PID in a Quadcopter.

**[Online Calculators and Tables to Help You Determine Proper Wire Size: ](http://wiresizecalculator.net/)** This article was used to get ideas about the wire sizes to be used for the project. 

**[(Part 3): This is how I created a self driving vehicle using Ai and Python: ](https://www.youtube.com/watch?v=n0RhimFSIDw&t=331s)** This video was to understand another design approach to using a ML model. 

**[Potentiometer Operated Steering: ](http://www.onlinejournal.in/IJIRV3I4/093.pdf)** This research paper was used to study an alternative to gather training data. 

**[Benewake TF Luna Product Introduction: ](https://www.youtube.com/watch?v=iLO75LWxVBA)** This video was used to make the purchase decision of getting affortable Lidar and how to use it. 

**[Autonomous Navigation, Part 2: Understanding the Particle Filter: ](https://www.youtube.com/watch?v=NrzmH_yerBU)** This video was used to understand how to make SLAM model with the help of a car. 

**[Raspberry Pi Essential Training: ](https://www.linkedin.com/learning/raspberry-pi-essential-training/transfering-the-exercise-files-to-the-pi?u=57694233)** This course was used to understand the basics of Raspberry Pi.

**[Install Tensorflow 2 on a Raspberry Pi 4 // Easy: ](https://www.youtube.com/watch?v=GNRg2P8Vqqs)** This video was used to install Tesorflow on Raspberry Pi

**[Installing Library package in Raspberry Pi-Chapter 2](https://www.pantechsolutions.net/blog/installing-library-packages-in-raspberry-pi/)** This article was used to install all other necessary libraries on Raspberry Pi.

**[Raspberry Pi 3: Extreme Passive Cooling: ](https://www.youtube.com/watch?v=1AYGnw6MwFM)** This video was used to evaluate different cooling solutions for raspberry pi.

**[Remote Access with SSH and Remote Desktop - Raspberry Pi and Python tutorials p.3: ](https://www.youtube.com/watch?v=IDqQIDL3LKg)** This video was used to set up SSH to control the raspberry pi as an alternative to VNC viwer.

**[Deep Neural Network Hardware Accelerator on FPGA Tutorial: ](https://www.youtube.com/watch?v=mA-b9qX1ySg)** This video was used to understand the implemetation of DNN on a FPGA

## Files Description

**[ModelApplication.py](/ModelApplication.py)** contains the implementaion of the Machine Learning Algorithm for steering. It also contains the communation with the GPIO pins of the Raspberry Pi 4 to control the wiper motor (steering motor) of our car. 

**[NVIDIA_Model.ipynb](/NVIDIA_Model.ipynb)**

![](/images/cnn-architecture.png)
[source](https://www.google.com/url?sa=i&url=https%3A%2F%2Fdeveloper.nvidia.com%2Fblog%2Fdeep-learning-self-driving-cars%2F&psig=AOvVaw0MfS5_e0LZlX-ZPg8U-KUy&ust=1617740053200000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCPitwLz15-8CFQAAAAAdAAAAABAD)

- This ML algorithm is written following the NVIDIA's model for CNN
- This code is also inspired from [this course](https://www.udemy.com/course/applied-deep-learningtm-the-complete-self-driving-car-course/) which has a great introduction to Machine Learning and also explains different ML types and algorithms 

**[PID.slx](/PID.slx)** contains the simulink model of the car

**[lanes.py](/lanes.py)** is a basic lane detection algorithm using Open CV and basic image processing

**[model.h5](/model.h5)** is the output file of the ML model. This file contains all the weights and biases of required to steer the car.

**[images](/images)** contains the images used for this file.

**[motorControl](/motorControl)** contains the sample code for controlling the Lidar sensor and the motor driver.
