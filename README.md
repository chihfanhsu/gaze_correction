Correcting gaze by warping-based convolutional neural network.

# system usage
```python
python regz_socket_MP_FD.py
```

# Setting configurations
edit config.py

# Parameters need to be personalized
Parameters "P_c_x", "P_c_y", "P_c_z", "S_W", and "S_H" need to be personalized. <br />
"P_c_x", "P_c_y", and "P_c_z": relative distance between the camera position and screen center (cm) <br />
"S_W" and "S_H": screen size (cm) <br />

# Starting to correct your gaze!
Push 'r' on the "local" window, and gaze your head on the "remote" window. <br />
Push 'q' on the "local" window to leave the program. <br />

# For multi-user
Parameters "tar_ip", "sender_port", and "recver_port" need to be defined. <br />
"tar_ip": target IP address <br />
"sender_port": port # for sending the redirected gaze video to the remote user <br />
"sender_port": port # for getting the redirected gaze video from the remote user <br />

# calibrating the focal length of camera
Execute the script "focal_length_calibration.ipynb" or "focal_length_calibration.py" and the estimated focal length (f) will be shown at top-left corner of window. <br />
Step 1: Please place your head in front of the camera about 50 cm <br />
Step 2: Please insert your interpupillary distance (the distance between two eyes) in the code or use the average value 6.3 cm <br />
![Calibration Example](https://github.com/chihfanhsu/gaze_correction/blob/master/gaze_correction_system/imgs/calibration.png)

# Implementation Environment
Python 3.5.3 <br />
Tensorflow 1.8.0

P.S. The video will delay at beginning because the TCP socket, and the video will be on time after few seconds.
