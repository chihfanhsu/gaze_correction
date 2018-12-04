Correcting gaze by warping-based convolutional neural network.

# system usage
```python
python regz_socket_MP_FD.py
```

# Setting configurations
edit config.py

# Parameters need to be personalized
Parameters "P_c_x", "P_c_y", "P_c_z", "S_W", and "S_H" need to be personalized.
"P_c_x", "P_c_y", and "P_c_z": relative distance between the camera position and screen center (cm)
"S_W" and "S_H": screen size (cm)

# Starting to correct your gaze!
Push 'r' on the "local" window, and gaze your head on the "remote" window.
Push 'q' on the "local" window to leave the program.

# For multi-user
Parameters "tar_ip", "sender_port", and "recver_port" need to be defined.
"tar_ip": target IP address
"sender_port": port # for sending the redirected gaze video to the remote user.
"sender_port": port # for getting the redirected gaze video from the remote user.

# calibrating the focal length of camera
Execute the script "focal_length_calibration.ipynb" or "focal_length_calibration.py" and the estimated focal length will be shown at top-left corner.
Step 1: Please place your head in front of the camera about 50 cm
Step 2: Please insert your interpupillary distance (the distance between two eyes) in the code or use the average value 6.3 cm

# Implementation Environment
Python 3.5.3
Tensorflow 1.8.0

P.S. The video will delay at beginning because the TCP socket, and the video will be on time after few seconds.
