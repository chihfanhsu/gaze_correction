
# coding: utf-8

# # Please place your head infront of the camera about 50 cm

# In[ ]:


d = 50


# # Please insert you Pupillary distance (distance between two eyes) or use the average value 6.3 cm

# In[ ]:


P_IPD = 6.3


# In[ ]:


import dlib
import cv2
import numpy as np


# In[ ]:


video_res = [640,480]
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./lm_feat/shape_predictor_68_face_landmarks.dat") 
face_detect_size = [320,240]

def get_eye_pos(shape, pos = "L"):
    if(pos == "R"):
        lc = 36
        rc = 39
        FP_seq = [36,37,38,39,40,41]
    elif(pos == "L"):
        lc = 42
        rc = 45
        FP_seq = [45,44,43,42,47,46]
    else:
        print("Error: Wrong Eye")

    eye_cx = (shape.part(rc).x+shape.part(lc).x)*0.5
    eye_cy = (shape.part(rc).y+shape.part(lc).y)*0.5
    eye_center = [eye_cx, eye_cy]
    eye_len = np.absolute(shape.part(rc).x - shape.part(lc).x)
    bx_d5w = eye_len*3/4
    bx_h = 1.5*bx_d5w
    sft_up = bx_h*7/12
    sft_low = bx_h*5/12
    E_TL = (int(eye_cx-bx_d5w),int(eye_cy-sft_up))
    E_RB = (int(eye_cx+bx_d5w),int(eye_cy+sft_low))
    return eye_center, E_TL, E_RB


# # Start capturing you faces, push k if you have already placed you head about 50 cm

# In[ ]:


vs = cv2.VideoCapture(0)

while 1:
    ret, recv_frame = vs.read()
    
    gray = cv2.cvtColor(recv_frame, cv2.COLOR_BGR2GRAY)
    face_detect_gray = cv2.resize(gray, (face_detect_size[0], face_detect_size[1]))
    # Detect the facial landmarks
    detections = detector(face_detect_gray, 0)
    x_ratio = video_res[0]/face_detect_size[0]
    y_ratio = video_res[1]/face_detect_size[1]
    LE_ach_maps=[]
    RE_ach_maps=[]
    for k,bx in enumerate(detections):
        target_bx = dlib.rectangle(left=int(bx.left()*x_ratio), right =int(bx.right()*x_ratio),
                                   top =int(bx.top()*y_ratio),  bottom=int(bx.bottom()*y_ratio))
        shape = predictor(gray, target_bx)
        # get eye
        LE_center, L_E_TL, L_E_RB = get_eye_pos(shape, pos="L")
        RE_center, R_E_TL, R_E_RB = get_eye_pos(shape, pos="R")

        f = int(np.sqrt((LE_center[0]-RE_center[0])**2 + (LE_center[1]-RE_center[1])**2)*d/P_IPD)
        cv2.rectangle(recv_frame,
                      (video_res[0]-150,0),(video_res[0],40),
                      (255,255,255),-1
                     )
        cv2.putText(recv_frame,
                    'f:'+str(f),
                    (video_res[0]-140,15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,0,255),1,cv2.LINE_AA)
#         cv2.line(recv_frame, (int(LE_center[0]),int(LE_center[1])), (int(RE_center[0]),int(RE_center[1])), (0,0,255))
        
        # eye region
        cv2.rectangle(recv_frame,
                      (L_E_TL[0],L_E_TL[1]),(L_E_RB[0],L_E_RB[1]),
                      (0,0,255),1
                     )
        cv2.rectangle(recv_frame,
                      (R_E_TL[0],R_E_TL[1]),(R_E_RB[0],R_E_RB[1]),
                      (0,0,255),1
                     )
        cv2.circle(recv_frame,(int(LE_center[0]),int(LE_center[1])), 2, (0,255,0), -1)
        cv2.circle(recv_frame,(int(RE_center[0]),int(RE_center[1])), 2, (0,255,0), -1)
        for i in range(68):
            cv2.circle(recv_frame,(shape.part(i).x,shape.part(i).y), 2, (0,0,255), -1)
    
    cv2.imshow("Calibration", recv_frame)
    k = cv2.waitKey(10)
    if k == ord('q'):
        vs.release()
        cv2.destroyAllWindows()
        break
    else:
        pass


# In[ ]:


print("focal length is ", f)

