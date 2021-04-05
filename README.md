# Self-driving Car
![](/images/pic01.jpg)

**Project Summary**

- Trained DNN to identify common traffic signs and trained a Machine Learning architecture based on The Nvidia Model to predict steering angles of a single laned road
- Utilized a lane detection algorthm to detect lanes and drive the car maintaining mid lane steering using PID control
- Equipped Raspberry Pi as the main processing unit and used it to communicate with Pi camera and motors
- Used Lidar to accurately detect nearby objects and take actions accordingly

## Files Description

[ModelApplication.py](/ModelApplication.py) contains the implementaion of the Machine Learning Algorithm for steering. It also contains the communation with the GPIO pins of the Raspberry Pi 4 to control the wiper motor (steering motor) of our car. 

[NVIDIA_Model.ipynb](/NVIDIA_Model.ipynb)
![](/images/cnn-architecture.png)
[source](https://www.google.com/url?sa=i&url=https%3A%2F%2Fdeveloper.nvidia.com%2Fblog%2Fdeep-learning-self-driving-cars%2F&psig=AOvVaw0MfS5_e0LZlX-ZPg8U-KUy&ust=1617740053200000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCPitwLz15-8CFQAAAAAdAAAAABAD)
- This ML algorithm is written following the NVIDIA's model for CNN
- This code is also inspired from [this course](https://www.udemy.com/course/applied-deep-learningtm-the-complete-self-driving-car-course/) which has a great introduction to Machine Learning and also explains different ML types and algorithms 
