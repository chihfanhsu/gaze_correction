Correcting gaze by warping-based convolutional neural network.
# Paper
@article{Hsu:2019:LMC:3339884.3311784,<br />
author = {Hsu, Chih-Fan and Wang, Yu-Shuen and Lei, Chin-Laung and Chen, Kuan-Ta},<br />
 title = {Look at Me\\\&Excl; Correcting Eye Gaze in Live Video Communication},<br />
 journal = {ACM Trans. Multimedia Comput. Commun. Appl.},<br />
 issue_date = {June 2019},<br />
 volume = {15},<br />
 number = {2},<br />
 month = jun,<br />
 year = {2019},<br />
 issn = {1551-6857},<br />
 pages = {38:1--38:21},<br />
 articleno = {38},<br />
 numpages = {21},<br />
 url = {http://doi.acm.org/10.1145/3311784},<br />
 doi = {10.1145/3311784},<br />
 acmid = {3311784},<br />
 publisher = {ACM},<br />
 address = {New York, NY, USA},<br />
 keywords = {Eye contact, convolutional neural network, gaze correction, <br />image processing, live video communication},<br />
} <br />

# Demo video on YouTube
[![Look at Me! Correcting Eye Gaze in Live Video Communication](https://github.com/chihfanhsu/gaze_correction/blob/master/imgs/YouTube_page.PNG)](https://youtu.be/9nAHINph5a4)

# System usage
```python
python regz_socket_MP_FD.py
```

# Parameters need to be personalized in the "config.py"
The positions of all parameters are illustrated in the following figure. P_o is the original point (0,0,0) which is defined at the center of the screen. <br />
<br />
Parameters "P_c_x", "P_c_y", "P_c_z", "S_W", "S_H", and "f" need to be personalized before using the system. <br /> 
"P_c_x", "P_c_y", and "P_c_z": relative distance between the camera position and screen center (cm) <br />
"S_W" and "S_H": screen size (cm) <br />
"f": focal length of camera <br />
<br />
![Parameters positions](https://github.com/chihfanhsu/gaze_correction/blob/master/imgs/correcting_gaze.png)

# Calibrating the focal length of the camera by the attached tools
Execute the script "focal_length_calibration.ipynb" or "focal_length_calibration.py" to estimated the focal length (f), and the value will be shown at the top-left corner of the window. <br />
Steps for calibration:<br />
Step 1, please place your head in front of the camera about 50 cm (you can change this value in the code) <br />
Step 2, please insert your interpupillary distance (the distance between two eyes) in the code or use the average value, 6.3 cm <br />
<br />
![Calibration Example](https://github.com/chihfanhsu/gaze_correction/blob/master/imgs/calibration.png)

# Starting to correct gaze! (Self-demo)
Push 'r' key when focusing the "local" window and gaze your head on the "remote" window to start gaze correction. <br />
Push 'q' key when focusing the "local" window to leave the program. <br />
<br />
*The video will delay at beginning because the TCP socket transmission, nevertheless, the video will be on time after few seconds. <br />
<br />
![System usage Example](https://github.com/chihfanhsu/gaze_correction/blob/master/imgs/system_usage.png)

# For online video communication
The codes at the local and remote sides are the same. However, parameters "tar_ip", "sender_port", and "recver_port" need to be defined at both sides. <br />
"tar_ip": the other user's IP address <br />
"sender_port": port # for sending the redirected gaze video to the other user <br />
"sender_port": port # for getting the redirected gaze video from the other user <br />

# IP setup for self-demo
The codes at the local and remote sides are the same. However, parameters "tar_ip", "sender_port", and "recver_port" need to be defined at both sides. <br />
"tar_ip": 127.0.0.1 <br />
"sender_port": 5005 <br />
"sender_port": 5005 <br />

# Environmental setup
Python 3.5.3 <br />
Tensorflow 1.8.0 <br />
Cuda V9.0.176 and corresponding cuDnn <br />

# Required packages
Dlib 18.17.100 <br />
OpenCV 3.4.1 <br />
Numpy 1.15.4 + mkl <br />
pypiwin32 <br />
scipy 0.19.1 <br />

# DIRL Gaze Dataset
![System usage Example](https://github.com/chihfanhsu/gaze_correction/blob/master/imgs/dataset_collection.PNG)
<br />
37 Asian volunteers participated in our dataset collection. About 100 gaze directions are collected in range +40 to -40 degrees in horizontal and +30 to -30 degrees in vertical, in which 63 and 37 images are fixed and random direction, respectively. The images with closed eyes were removed.
[Download here!](https://sites.google.com/site/chihfanhsuwebsite/dataset)

# Several Exciting Projects
[![2019 Eye Contact Correction using Deep Neural Networks]](https://arxiv.org/pdf/1906.05378.pdf) <br />
[![2019 Photo-Realistic Monocular Gaze Redirection Using Generative Adversarial Networks]](https://arxiv.org/pdf/1903.12530.pdf)