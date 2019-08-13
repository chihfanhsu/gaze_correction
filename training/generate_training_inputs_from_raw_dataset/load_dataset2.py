import threading
import numpy as np
import tensorflow as tf
import pickle
import cv2
import os

sur_agl_limit = 16
dif_agl_limit_v = 21
dif_agl_limit_h = 26

def seperate_eye_and_lid(imgs, anchor_maps):
    # start_time = time.time()
    imgs_eye = []
    imgs_lid = []
    imgs_eye_mask = []
    imgs_lid_mask = []
    for user_idx in range(len(imgs)):
        user_eye = []
        user_lid = []
        user_eye_mask = []
        user_lid_mask = []
        for img_idx in range(anchor_maps[user_idx].shape[0]):
            # get eye anchors
            anchors = []
            for i in range(6):
                anchor = [int(np.where(anchor_maps[user_idx][img_idx,0,:,2*i+0] ==0)[0]),int(np.where(anchor_maps[user_idx][img_idx,:,0,2*i+1] ==0)[0])]
                anchors.append(anchor)
            anchors = np.array(anchors, np.int32)
            # create mask
            mask_eye = np.zeros((imgs[0].shape[1],imgs[0].shape[2]),np.uint8)
            cv2.fillPoly(mask_eye, [anchors], 1)
            mask_lid = np.ones((imgs[0].shape[1],imgs[0].shape[2]),np.uint8) - mask_eye
            # crop imaage
            #temp = cv2.cvtColor(mask_lid,cv2.COLOR_GRAY2RGB)
            img_eye = cv2.bitwise_and(imgs[user_idx][img_idx,...],imgs[user_idx][img_idx,...],mask = mask_eye)# + temp
            img_lid = cv2.bitwise_and(imgs[user_idx][img_idx,...],imgs[user_idx][img_idx,...],mask = mask_lid)
            user_eye.append(img_eye)
            user_lid.append(img_lid)
            user_eye_mask.append(mask_eye)
            user_lid_mask.append(mask_lid)
        user_eye = np.array(user_eye)
        user_lid = np.array(user_lid)
        user_eye_mask = np.array(user_eye_mask)
        user_lid_mask = np.array(user_lid_mask)
        imgs_eye.append(user_eye)
        imgs_lid.append(user_lid)
        imgs_eye_mask.append(user_eye_mask)
        imgs_lid_mask.append(user_lid_mask)
    # print("sep time %.4f" % (time.time()-start_time))
    return imgs_eye, imgs_lid, imgs_eye_mask, imgs_lid_mask

### define your input schedual #####
def read_training_data(file_path):
    f = open(file_path, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def load(data_dir, dirs, eye):
    MyDict = {}
    imgs = []
    agls = []
    ps = []
    anchor_maps = []
    for d in dirs:
        print(os.path.join(data_dir, d))
        data = read_training_data(os.path.join(data_dir, d) + '/' + d + str('_') + eye)
        imgs.append(np.asarray(data['img'], dtype= np.float32)/255.0)            
        agls.append(np.concatenate([np.expand_dims(np.asarray(data['v'], dtype= np.float32), axis=1),
                                                   np.expand_dims(np.asarray(data['h'], dtype= np.float32), axis=1)],
                                    axis = 1))
        ps.append(np.asarray(data['p'], dtype= np.float32))
        anchor_maps.append(np.asarray(data['anchor_map'], dtype= np.float32))
    # sep image to eye and lid
    imgs_eye, imgs_lid, msk_eye, msk_lid = seperate_eye_and_lid(imgs, anchor_maps)
        
    MyDict['imgs_ori'] = imgs
    MyDict['agls'] = agls
    MyDict['ps'] = ps
    MyDict['anchor_maps'] = anchor_maps
    MyDict['imgs_eye'] = imgs_eye
    MyDict['imgs_lid'] = imgs_lid
    MyDict['msk_eye'] = msk_eye
    MyDict['msk_lid'] = msk_lid
    return MyDict

def img_pair_list(agls, pose):
    '''
    10 dims:
    pose, uid,
    src_img_idx,tar_img_idx,
    src_v,src_h,
    tar_v,tar_h,
    agl_dif_v,agl_dif_h
    '''
    for uid in range(len(agls)):
        n_agl = np.arange(len(agls[uid]))
        sur, tar = np.meshgrid(n_agl, n_agl)
        uid_pair = np.concatenate((np.expand_dims(np.repeat(pose, len(agls[uid])*len(agls[uid])), axis=1),
                                   np.expand_dims(np.repeat(uid, len(agls[uid])*len(agls[uid])), axis=1),
                                   np.expand_dims(np.reshape(sur,-1), axis=1),
                                   np.expand_dims(np.reshape(tar,-1), axis=1)), axis=1)
        if uid == 0:
            pairs = uid_pair
            src_agls = agls[uid][uid_pair[:,2],:]
            tar_agls = agls[uid][uid_pair[:,3],:]
            dif_agls = agls[uid][uid_pair[:,3],:] - agls[uid][uid_pair[:,2],:]
        else:
            pairs = np.concatenate((pairs, uid_pair), axis=0) # image index
            src_agls = np.concatenate((src_agls, agls[uid][uid_pair[:,2],:]), axis=0) # sourse angle
            tar_agls = np.concatenate((tar_agls, agls[uid][uid_pair[:,3],:]), axis=0)
            dif_agls = np.concatenate((dif_agls, agls[uid][uid_pair[:,3],:] - agls[uid][uid_pair[:,2],:]), axis=0)

    pairs = np.concatenate((pairs,src_agls,tar_agls,dif_agls), axis=1)           
    return pairs.astype(np.int32)

def data_iterator(input_dict, pairs, batch_size, shuffle = True):
    # print(input_dict.keys())
    t_batch = int(len(pairs)/batch_size)
    
    while True:
        idxs = np.arange(0, len(pairs))
        if(shuffle):
            np.random.shuffle(idxs)
            
        for batch_idx in range(t_batch-1):
            cur_idxs = idxs[(batch_idx*batch_size):((batch_idx+1)*batch_size)]
            pairs_batch = pairs[cur_idxs]
            out_dict = {}
            b_pose =[]
            b_uID =[]
            b_img_ori = []
            b_sur_agl = []
            b_tar_agl = []
            b_fp = []
            b_img__ori = []
            b_msk_eye = []
            for pair_idx in range(len(pairs_batch)):
                pose = str(pairs_batch[pair_idx,0])
                uID = pairs_batch[pair_idx,1]
                surID = pairs_batch[pair_idx,2]
                tarID = pairs_batch[pair_idx,3]
                b_pose.append(pose)
                b_uID.append(uID)
                b_img_ori.append(input_dict[pose]['imgs_ori'][uID][surID])
                b_sur_agl.append(input_dict[pose]['agls'][uID][surID])
                b_tar_agl.append(input_dict[pose]['agls'][uID][tarID])
                b_fp.append(input_dict[pose]['anchor_maps'][uID][surID])                
                b_img__ori.append(input_dict[pose]['imgs_ori'][uID][tarID])                
                b_msk_eye.append(input_dict[pose]['msk_eye'][uID][surID])
            out_dict['pose'] = np.asarray(b_pose)
            out_dict['uID'] = np.asarray(b_uID)
            out_dict['imgs_ori'] = np.asarray(b_img_ori)
            out_dict['fp'] = np.asarray(b_fp)
            out_dict['sur_agl'] = np.asarray(b_sur_agl)
            out_dict['tar_agl'] = np.asarray(b_tar_agl)
            out_dict['imgs__ori'] = np.asarray(b_img__ori)
            out_dict['msk_eye'] = np.asarray(b_msk_eye)
            
            yield out_dict

def shuffle_data_batch(data_batch):
    idxs = np.arange(0, data_batch['imgs_ori'].shape[0])
    np.random.shuffle(idxs)
    for i in data_batch.keys():
        # print(data_batch.keys())
        data_batch[i] = data_batch[i][idxs,...]
    return data_batch

def get_dict(conf, tar_path):
    tar_dir = os.path.join(conf.data_dir, conf.dataset, tar_path)
    pose_dirs = np.asarray([d for d in os.listdir(tar_dir) if os.path.isdir(os.path.join(tar_dir, d))])
    print("Pose dirs", pose_dirs)
    tar_dicts = {}
    for p in pose_dirs:
        print('pose', p)
        tar_dirs = np.asarray([d for d in os.listdir(os.path.join(tar_dir, p)) if os.path.isdir(os.path.join(tar_dir, p, d))])
        print("Dirs", tar_dirs)
        # load training inputs
        tar_dict = load(data_dir=os.path.join(tar_dir, p), dirs = tar_dirs, eye = conf.eye)
        tar_dict['pairs'] = img_pair_list(tar_dict['agls'], int(p))
        tar_dicts[p] = tar_dict
    
    return tar_dicts

def get_easy_hard_iter(input_dicts, batch_size):
    sur_agl_limit = 16
    dif_agl_limit_v = 21
    dif_agl_limit_h = 26
    pairs = []
    for pose in list(input_dicts.keys()):
        if pose == list(input_dicts.keys())[0]:
            pairs = input_dicts[pose]['pairs']
        else:
            pairs = np.concatenate((pairs, input_dicts[pose]['pairs']), axis = 0) # image index

    all_idx = np.arange(len(pairs))
    easy_idx = np.where((np.abs(pairs[:,4]) < sur_agl_limit) &
                                      (np.abs(pairs[:,5]) < sur_agl_limit) &
                                      (np.abs(pairs[:,8]) < dif_agl_limit_v) &
                                      (np.abs(pairs[:,9]) < dif_agl_limit_h))[0]

    hard_idx = np.setdiff1d(all_idx, easy_idx)
    if (len(all_idx) != (len(hard_idx) + len(easy_idx))):
        sys.exit("[T] Easy and Hard sets separation error")

    print("E {}; H {}; ALL {}".format(len(easy_idx),len(hard_idx),len(all_idx)))
    easy_iter_ = data_iterator(input_dicts, pairs[easy_idx,:], batch_size)
    hard_iter_ = data_iterator(input_dicts, pairs[hard_idx,:], batch_size)
    
    return easy_iter_, hard_iter_, len(easy_idx),len(hard_idx)

def merge_batches(cur_batch, tar_batch):
    for i in cur_batch.keys():
        cur_batch[i] = np.concatenate((cur_batch[i], tar_batch[i]), axis=0)
    return cur_batch
