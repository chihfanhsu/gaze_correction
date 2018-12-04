
# coding: utf-8

# # load package and settings

# In[ ]:


import cv2
import sys
import dlib
import time
import socket
import struct
import numpy as np
import tensorflow as tf
from win32api import GetSystemMetrics
import win32gui

from threading import Thread, Lock
import multiprocessing as mp
from config import get_config
import pickle
import math


# In[ ]:


conf,_ = get_config()
if conf.mod == 'flx':
    import flx as model
else:
    sys.exit("Wrong Model selection: flx or deepwarp")

# system parameters
model_dir = './'+conf.weight_set+'/warping_model/'+conf.mod+'/'+ str(conf.ef_dim) + '/'
size_video = [640,480]
# fps = 0
P_IDP = 5
depth = -50
# for monitoring

# environment parameter
Rs = (GetSystemMetrics(0),GetSystemMetrics(1))


# In[ ]:


model_dir
print(Rs)


# In[ ]:


# video receiver
class video_receiver:
    def __init__(self,shared_v,lock):
        self.close = False
        self.video_recv = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        print('Socket created')
        #         global remote_head_Center
        self.video_recv.bind(('',conf.recver_port))
        self.video_recv.listen(10)
        print('Socket now listening')
        self.conn, self.addr=self.video_recv.accept()
        # face detection
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./lm_feat/shape_predictor_68_face_landmarks.dat") 
        self.face_detect_size = [320,240]
        self.x_ratio = size_video[0]/self.face_detect_size[0]
        self.y_ratio = size_video[1]/self.face_detect_size[1]      
        self.start_recv(shared_v,lock)

    def face_detection(self,frame,shared_v,lock):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detect_gray = cv2.resize(gray,(self.face_detect_size[0],self.face_detect_size[1]))
        detections = self.detector(face_detect_gray, 0)
        coor_remote_head_center=[0,0]
        for k,bx in enumerate(detections):
            coor_remote_head_center = [int((bx.left()+bx.right())*self.x_ratio/2),
                                       int((bx.top()+bx.bottom())*self.y_ratio/2)]
            break
        # share remote participant's eye to the main process
        lock.acquire()
        shared_v[0] = coor_remote_head_center[0]
        shared_v[1] = coor_remote_head_center[1]
        lock.release()

    def start_recv(self,shared_v,lock):
        data = b""
        payload_size = struct.calcsize("L")
        print("payload_size: {}".format(payload_size))
        while True:
            while len(data) < payload_size:
                data += self.conn.recv(4096)

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]
            while len(data) < msg_size:
                data += self.conn.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]
            frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
            if frame == 'stop':
                print('stop')
                cv2.destroyWindow("Remote")
                break
            
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            
            # face detection
            self.video_recv_hd_thread = Thread(target=self.face_detection, args=(frame,shared_v,lock))
            self.video_recv_hd_thread.start()
        
            cv2.imshow('Remote',frame)
            cv2.waitKey(1)


# # Flx-gaze 

# In[ ]:


class gaze_redirection_system:
    def __init__(self,shared_v,lock):
        #Landmark identifier. Set the filename to whatever you named the downloaded file
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./lm_feat/shape_predictor_68_face_landmarks.dat") 
        self.size_df = (320,240)
        self.size_I = (48,64)
        # initial value
        self.Rw = [0,0]
        self.Pe_z = -60
        #### get configurations
        self.f = conf.f
        self.Ps = (conf.S_W,conf.S_H)
        self.Pc = (conf.P_c_x,conf.P_c_y,conf.P_c_z)
        self.Pe = [self.Pc[0],self.Pc[1],self.Pe_z] # H,V,D
        ## start video sender
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.client_socket.connect((conf.tar_ip, conf.sender_port))
        self.encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
        
        # load model to gpu
        print("Loading model of [L] eye to GPU")
        with tf.Graph().as_default() as g:
            # define placeholder for inputs to network
            with tf.name_scope('inputs'):
                self.LE_input_img = tf.placeholder(tf.float32, [None, conf.height, conf.width, conf.channel], name="input_img")
                self.LE_input_fp = tf.placeholder(tf.float32, [None, conf.height, conf.width,conf.ef_dim], name="input_fp")
                self.LE_input_ang = tf.placeholder(tf.float32, [None, conf.agl_dim], name="input_ang")
                self.LE_phase_train = tf.placeholder(tf.bool, name='phase_train') # a bool for batch_normalization

            self.LE_img_pred, _, _ = model.inference(self.LE_input_img, self.LE_input_fp, self.LE_input_ang, self.LE_phase_train, conf)

            # split modle here
            self.L_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False), graph = g)
            # load model
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(model_dir+'L/')
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(self.L_sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')

        print("Loading model of [R] eye to GPU")
        with tf.Graph().as_default() as g2:
            # define placeholder for inputs to network
            with tf.name_scope('inputs'):
                self.RE_input_img = tf.placeholder(tf.float32, [None, conf.height, conf.width, conf.channel], name="input_img")
                self.RE_input_fp = tf.placeholder(tf.float32, [None, conf.height, conf.width,conf.ef_dim], name="input_fp")
                self.RE_input_ang = tf.placeholder(tf.float32, [None, conf.agl_dim], name="input_ang")
                self.RE_phase_train = tf.placeholder(tf.bool, name='phase_train') # a bool for batch_normalization

            self.RE_img_pred, _, _ = model.inference(self.RE_input_img, self.RE_input_fp, self.RE_input_ang, self.RE_phase_train, conf)

            # split modle here
            self.R_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False), graph = g2)
            # load model
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(model_dir+'R/')
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(self.R_sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
               
        self.run(shared_v,lock)
        
    def monitor_para(self,frame,fig_alpha,fig_eye_pos,fig_R_w):
        cv2.rectangle(frame,
                  (size_video[0]-150,0),(size_video[0],55),
                  (255,255,255),-1
                 )
        cv2.putText(frame,
                    'Eye:['+str(int(fig_eye_pos[0])) +','+str(int(fig_eye_pos[1]))+','+str(int(fig_eye_pos[2]))+']',
                    (size_video[0]-140,15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,0,255),1,cv2.LINE_AA)
        cv2.putText(frame,
                    'alpha:[V='+str(int(fig_alpha[0])) + ',H='+ str(int(fig_alpha[1]))+']',
                    (size_video[0]-140,30),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1,cv2.LINE_AA)
        cv2.putText(frame,
                    'R_w:['+str(int(fig_R_w[0])) + ','+ str(int(fig_R_w[1]))+']',
                    (size_video[0]-140,45),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1,cv2.LINE_AA)
        return frame
        
    def get_inputs(self, frame, shape, pos = "L", size_I = [48,64]):
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
        img_eye = frame[int(eye_cy-sft_up):int(eye_cy+sft_low),int(eye_cx-bx_d5w):int(eye_cx+bx_d5w)]
        ori_size = [img_eye.shape[0],img_eye.shape[1]]
        LT_coor = [int(eye_cy-sft_up), int(eye_cx-bx_d5w)] # (y,x)    
        img_eye = cv2.resize(img_eye, (size_I[1],size_I[0]))
        # create anchor maps
        ach_map = []
        for i,d in enumerate(FP_seq):
            resize_x = int((shape.part(d).x-LT_coor[1])*size_I[1]/ori_size[1])
            resize_y = int((shape.part(d).y-LT_coor[0])*size_I[0]/ori_size[0])
            # y
            ach_map_y = np.expand_dims(np.expand_dims(np.arange(0, size_I[0]) - resize_y, axis=1), axis=2)
            ach_map_y = np.tile(ach_map_y, [1,size_I[1],1])
            # x
            ach_map_x = np.expand_dims(np.expand_dims(np.arange(0, size_I[1]) - resize_x, axis=0), axis=2)
            ach_map_x = np.tile(ach_map_x, [size_I[0],1,1])
            if (i ==0):
                ach_map = np.concatenate((ach_map_x, ach_map_y), axis=2)
            else:
                ach_map = np.concatenate((ach_map, ach_map_x, ach_map_y), axis=2)

        return img_eye/255, ach_map, eye_center, ori_size, LT_coor
       
    def shifting_angles_estimator(self, R_le, R_re,shared_v,lock):
        # get P_w
        try:
            tar_win = win32gui.FindWindow(None, "Remote")
            #left, top, reight, bottom
            Rw_lt = win32gui.GetWindowRect(tar_win)
            size_window = (Rw_lt[2]-Rw_lt[0], Rw_lt[3]-Rw_lt[1])
        except:
            Rw_lt = [int(Rs[0])-int(size_window[0]/2),int(Rs[1])-int(size_window[1]/2)]
            size_window = (659,528)
            print("Missing the window")
        # get pos head
        pos_remote_head = [int(size_window[0]/2),int(size_window[1]/2)]
        
        try:
            if ((shared_v[0] !=0) & (shared_v[1] !=0)):
                pos_remote_head[0] = shared_v[0]
                pos_remote_head[1] = shared_v[1]

        except:
            pos_remote_head = (int(size_window[0]/2),int(size_window[1]/2))
            
        R_w = (Rw_lt[0]+pos_remote_head[0], Rw_lt[1]+pos_remote_head[1])
        Pw = (self.Ps[0]*(R_w[0]-Rs[0]/2)/Rs[0], self.Ps[1]*(R_w[1]-Rs[1]/2)/Rs[1], 0)

        # get Pe
        self.Pe[2] = -(self.f*conf.P_IDP)/np.sqrt((R_le[0]-R_re[0])**2 + (R_le[1]-R_re[1])**2)
        # x-axis needs flip
        self.Pe[0] = -np.abs(self.Pe[2])*(R_le[0]+R_re[0]-size_video[0])/(2*self.f) + self.Pc[0]
        self.Pe[1] = np.abs(self.Pe[2])*(R_le[1]+R_re[1]-size_video[1])/(2*self.f) + self.Pc[1]

        # calcualte alpha
        a_w2z_x = math.degrees(math.atan( (Pw[0]-self.Pe[0])/(Pw[2]-self.Pe[2])))
        a_w2z_y = math.degrees(math.atan( (Pw[1]-self.Pe[1])/(Pw[2]-self.Pe[2])))    

        a_z2c_x = math.degrees(math.atan( (self.Pe[0]-self.Pc[0])/(self.Pc[2]-self.Pe[2])))
        a_z2c_y = math.degrees(math.atan( (self.Pe[1]-self.Pc[1])/(self.Pc[2]-self.Pe[2])))

        alpha = [int(a_w2z_y + a_z2c_y),int(a_w2z_x + a_z2c_x)] # (V,H)

        return alpha, self.Pe, R_w
    
    def flx_gaze(self, frame, gray, detections, shared_v, lock, pixel_cut=[3,4], size_I = [48,64]):
        alpha_w2c = [0,0]
        x_ratio = size_video[0]/self.size_df[0]
        y_ratio = size_video[1]/self.size_df[1]
        LE_M_A=[]
        RE_M_A=[]
        p_e=[0,0]
        R_w=[0,0]
        for k,bx in enumerate(detections):
            # Get facial landmarks
            time_start = time.time()
            target_bx = dlib.rectangle(left=int(bx.left()*x_ratio),right =int(bx.right()*x_ratio),
                                       top =int(bx.top()*y_ratio), bottom=int(bx.bottom()*y_ratio))
            shape = self.predictor(gray, target_bx)
            # get eye
            LE_img, LE_M_A, LE_center, size_le_ori, R_le_LT = self.get_inputs(frame, shape, pos="L", size_I=size_I)
            RE_img, RE_M_A, RE_center, size_re_ori, R_re_LT = self.get_inputs(frame, shape, pos="R", size_I=size_I)
            # shifting angles estimator
            alpha_w2c, p_e, R_w = self.shifting_angles_estimator(LE_center,RE_center,shared_v,lock)
            
            time_get_eye = time.time() - time_start
            # gaze manipulation
            time_start = time.time()
            
            # gaze redirection
            # left Eye
            LE_infer_img = self.L_sess.run(self.LE_img_pred, feed_dict= {
                                                            self.LE_input_img: np.expand_dims(LE_img, axis = 0),
                                                            self.LE_input_fp: np.expand_dims(LE_M_A, axis = 0),
                                                            self.LE_input_ang: np.expand_dims(alpha_w2c, axis = 0),
                                                            self.LE_phase_train: False
                                                         })
            LE_infer = cv2.resize(LE_infer_img.reshape(size_I[0],size_I[1],3), (size_le_ori[1], size_le_ori[0]))
            # right Eye
            RE_infer_img = self.R_sess.run(self.RE_img_pred, feed_dict= {
                                                            self.RE_input_img: np.expand_dims(RE_img, axis = 0),
                                                            self.RE_input_fp: np.expand_dims(RE_M_A, axis = 0),
                                                            self.RE_input_ang: np.expand_dims(alpha_w2c, axis = 0),
                                                            self.RE_phase_train: False
                                                         })
            RE_infer = cv2.resize(RE_infer_img.reshape(size_I[0],size_I[1],3), (size_re_ori[1], size_re_ori[0]))
            
            # repace eyes
            frame[(R_le_LT[0]+pixel_cut[0]):(R_le_LT[0]+size_le_ori[0]-pixel_cut[0]),
                  (R_le_LT[1]+pixel_cut[1]):(R_le_LT[1]+size_le_ori[1]-pixel_cut[1])] = LE_infer[pixel_cut[0]:(-1*pixel_cut[0]), pixel_cut[1]:-1*(pixel_cut[1])]*255
            frame[(R_re_LT[0]+pixel_cut[0]):(R_re_LT[0]+size_re_ori[0]-pixel_cut[0]),
                  (R_re_LT[1]+pixel_cut[1]):(R_re_LT[1]+size_re_ori[1]-pixel_cut[1])] = RE_infer[pixel_cut[0]:(-1*pixel_cut[0]), pixel_cut[1]:-1*(pixel_cut[1])]*255

        frame = self.monitor_para(frame, alpha_w2c, self.Pe, R_w)

        result, imgencode = cv2.imencode('.jpg', frame, self.encode_param)
        data = pickle.dumps(imgencode, 0)
        self.client_socket.sendall(struct.pack("L", len(data)) + data)
        return True
        
    def redirect_gaze(self, frame,shared_v,lock):
        # head detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detect_gray = cv2.resize(gray,(self.size_df[0],self.size_df[1]))
        detections = self.detector(face_detect_gray, 0)
           
        rg_thread = Thread(target=self.flx_gaze, args=(frame, gray, detections,shared_v,lock))
        rg_thread.start()
        return True
    
    def run(self,shared_v,lock):
        # def main():
        redir = False
        size_window = [659,528]
        vs = cv2.VideoCapture(0)
        vs.set(3, size_video[0])
        vs.set(4, size_video[1])
        t = time.time()
        cv2.namedWindow(conf.uid)
        cv2.moveWindow(conf.uid, int(Rs[0]/2)-int(size_window[0]/2),int(Rs[1]/2)-int(size_window[1]/2));
        while 1:
            ret, recv_frame = vs.read()
            if ret:
                cv2.imshow(conf.uid,recv_frame)
                if recv_frame is not None:
                    # redirected gaze
                    if redir:
                        frame = recv_frame.copy()
                        try:
                            tag = self.redirect_gaze(frame,shared_v,lock)
                        except:
                            pass
                    else:
                        result, imgencode = cv2.imencode('.jpg', recv_frame, self.encode_param)
                        data = pickle.dumps(imgencode, 0)
                        self.client_socket.sendall(struct.pack("L", len(data)) + data)

                    if (time.time() - t) > 1:
                        t = time.time()

                    k = cv2.waitKey(10)
                    if k == ord('q'):
                        data = pickle.dumps('stop')
                        self.client_socket.sendall(struct.pack("L", len(data))+data)
                        time.sleep(3)
                        cv2.destroyWindow(conf.uid)
                        self.client_socket.shutdown(socket.SHUT_RDWR)
                        self.client_socket.close()
                        vs.release()
                        self.L_sess.close()
                        self.R_sess.close()
                        break
                    elif k == ord('r'):
                        if redir:
                            redir = False
                        else:
                            redir = True
                    else:
                        pass


# In[ ]:


if __name__ == '__main__':
    l = mp.Lock()  # multi-process lock
    v = mp.Array('i', [320,240])  # shared parameter
    # start video receiver
    # vs_thread = Thread(target=video_receiver, args=(conf.recver_port,))
    vs_thread = mp.Process(target=video_receiver, args=(v,l))
    vs_thread.start()
    time.sleep(1)
    gz_thread = mp.Process(target=gaze_redirection_system, args=(v,l))
    gz_thread.start()
    vs_thread.join()
    gz_thread.join()

