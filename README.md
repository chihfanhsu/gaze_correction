Correcting gaze by warping-based convolutional neural network.

# System usage
```python
python regz_socket_MP_FD.py
```
# Parameters need to be personalized in the "config.py"
The positions of all parameters are illustrated in the following figure. P_o is the original point (0,0,0) which is defind at the center of the screen. <br />
<br />
Parameters "P_c_x", "P_c_y", "P_c_z", "S_W", "S_H", and "f" need to be personalized before using the system. <br /> 
"P_c_x", "P_c_y", and "P_c_z": relative distance between the camera position and screen center (cm) <br />
"S_W" and "S_H": screen size (cm) <br />
"f": focal length of camera <br />
<br />
![Parameters positions](https://github.com/chihfanhsu/gaze_correction/blob/master/imgs/correcting_gaze.png)

# Calibrating the focal length of camera by the attached tools
Execute the script "focal_length_calibration.ipynb" or "focal_length_calibration.py" to estimated the focal length (f), and the value will be shown at top-left corner of window. <br />
Steps for caligration:
Step 1, please place your head in front of the camera about 50 cm (you can change this value in the code) <br />
Step 2, please insert your interpupillary distance (the distance between two eyes) in the code or use the average value 6.3 cm <br />
<br />
![Calibration Example](https://github.com/chihfanhsu/gaze_correction/blob/master/imgs/calibration.png)

# Starting to correct gaze!
Push 'r' key when focusing the "local" window and gaze your head on the "remote" window to start gaze correction. <br />
Push 'q' key when focusing the "local" window to leave the program. <br />
<br />
*The video will delay at beginning because the TCP socket, and the video will be on time after few seconds. <br />
<br />
![System usage Example](https://github.com/chihfanhsu/gaze_correction/blob/master/imgs/system_usage.png)

# For multi-user
The codes at the local and remote sides are the same. However, parameters "tar_ip", "sender_port", and "recver_port" need to be defined at both side. <br />
"tar_ip": the other user's IP address <br />
"sender_port": port # for sending the redirected gaze video to the other user <br />
"sender_port": port # for getting the redirected gaze video from the other user <br />

# Implementation Environment
Python 3.5.3 <br />
Tensorflow 1.8.0 <br />
