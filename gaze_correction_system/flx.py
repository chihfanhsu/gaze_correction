import tensorflow as tf
import numpy as np
import tf_utils
import transformation

img_crop = 3

def gen_agl_map(inputs, height, width,feature_dims):
    with tf.name_scope("gen_agl_map"):
        batch_size = tf.shape(inputs)[0]
        ret = tf.reshape(tf.tile(inputs,tf.constant([1,height*width])), [batch_size,height,width,feature_dims])
        return ret

def encoder(inputs, height, width, tar_dim):
    with tf.variable_scope('encoder'):
        dnn_blk_0 =  tf_utils.dnn_blk(inputs, 16, name = 'dnn_blk_0')
        dnn_blk_1 =  tf_utils.dnn_blk(dnn_blk_0, 16, name = 'dnn_blk_1')
        dnn_blk_2 =  tf_utils.dnn_blk(dnn_blk_1, tar_dim, name = 'dnn_blk_2')
        agl_map = gen_agl_map(dnn_blk_2, height, width, tar_dim)
        return agl_map

def apply_lcm(batch_img, light_weight):
    with tf.name_scope('apply_lcm'):
        img_wgts, pal_wgts = tf.split(light_weight, [1,1], 3)
        img_wgts = tf.tile(img_wgts, [1,1,1,3])
        pal_wgts = tf.tile(pal_wgts, [1,1,1,3])
        palette = tf.ones(tf.shape(batch_img), dtype = tf.float32)
        ret = tf.add(tf.multiply(batch_img, img_wgts), tf.multiply(palette, pal_wgts))
        return ret
    
def trans_module(inputs, structures, phase_train, name="trans_module"):
    with tf.variable_scope(name) as scope:
        cnn_blk_0 = tf_utils.cnn_blk(inputs, structures['depth'][0], structures['filter_size'][0], phase_train, name = 'cnn_blk_0')
        cnn_blk_1 = tf_utils.cnn_blk(cnn_blk_0, structures['depth'][1], structures['filter_size'][1], phase_train, name = 'cnn_blk_1')
        cnn_blk_2 = tf_utils.cnn_blk(tf.concat([cnn_blk_0,cnn_blk_1], axis=3), structures['depth'][2], structures['filter_size'][2], phase_train, name = 'cnn_blk_2')
        cnn_blk_3 = tf_utils.cnn_blk(tf.concat([cnn_blk_0,cnn_blk_1,cnn_blk_2], axis=3), structures['depth'][3], structures['filter_size'][3], phase_train, name = 'cnn_blk_3')
        cnn_4 = tf.layers.conv2d(inputs=cnn_blk_3, filters=structures['depth'][4], kernel_size=structures['filter_size'][4], padding="same", activation=None, use_bias=False, name="cnn_4")
        return cnn_4
    
def lcm_module(inputs, structures, phase_train, name="lcm_module"):
    with tf.variable_scope(name) as scope:
        cnn_blk_0 = tf_utils.cnn_blk(inputs, structures['depth'][0], structures['filter_size'][0], phase_train, name = 'cnn_blk_0')        
        cnn_blk_1 = tf_utils.cnn_blk(cnn_blk_0, structures['depth'][1], structures['filter_size'][1], phase_train, name = 'cnn_blk_1')
        cnn_2 = tf.layers.conv2d(inputs=cnn_blk_1, filters=structures['depth'][2], kernel_size=structures['filter_size'][2], padding="same", activation=None, use_bias=False, name='cnn_2')
        lcm_map = tf.nn.softmax(cnn_2)
        return lcm_map

def inference(input_img, input_fp, input_agl, phase_train, conf):
    """Build the Deepwarp model.
    Args: images, anchors_map of eye, angle 
    Returns: lcm images
    """
    corse_layer = {'depth':(32,64,64,32,16), 'filter_size':([5,5],[3,3],[3,3],[3,3],[1,1])}
    fine_layer = {'depth':(32,64,32,16,4), 'filter_size':([5,5],[3,3],[3,3],[3,3],[1,1])}
    lcm_layer = {'depth':(8,8,2), 'filter_size':([3,3],[3,3],[1,1])}
    
    with tf.variable_scope('warping_model'):        
        agl_map = encoder(input_agl, conf.height, conf.width, conf.encoded_agl_dim)        
        igt_inputs = tf.concat([input_img, input_fp, agl_map],axis=3)
        
        with tf.variable_scope('warping_module'):
            '''coarse module'''
            resized_igt_inputs = tf.layers.average_pooling2d(inputs=igt_inputs, pool_size=[2,2], strides=2, padding='same')
            cours_raw = trans_module(resized_igt_inputs, corse_layer, phase_train, name='coarse_level')
            cours_act = tf.nn.tanh(cours_raw)
            coarse_resize = tf.image.resize_images(cours_act, (conf.height, conf.width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            coarse_out = tf.layers.average_pooling2d(inputs=coarse_resize, pool_size=[2, 2], strides=1, padding='same')
            '''fine module'''
            fine_input = tf.concat([igt_inputs, coarse_out],axis=3, name='fine_input')
            fine_out = trans_module(fine_input, fine_layer, phase_train, name='fine_level')
            flow_raw, lcm_input = tf.split(fine_out, [2,2], 3)

        flow = tf.nn.tanh(flow_raw)
        cfw_img = transformation.apply_transformation(flows = flow, img = input_img, num_channels=3)
        '''lcm module'''
        lcm_map = lcm_module(lcm_input, lcm_layer, phase_train, name="lcm_module")
        img_pred = apply_lcm(batch_img=cfw_img, light_weight=lcm_map)

        return img_pred, flow_raw, lcm_map

def dist_loss(y_pred, y_, method="MAE"):
    with tf.variable_scope('img_dist_loss'):
        loss = 0
        if(method == "L2"):
            loss = tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_), axis=3, keep_dims = True))
        elif (method == "MAE"):
            loss = tf.abs(y_pred - y_)
        loss = loss[:,img_crop:(-1)*img_crop,img_crop:(-1)*img_crop,:]
        loss = tf.reduce_sum(loss, axis = [1,2,3])
        return tf.reduce_mean(loss, axis=0)

def TVloss(inputs):
    with tf.variable_scope('TVloss'):
        dinputs_dx = inputs[:, :-1, :, :] - inputs[:, 1:, :, :]
        dinputs_dy = inputs[:, :, :-1, :] - inputs[:, :, 1:, :]
        dinputs_dx = tf.pad(dinputs_dx, [[0,0],[0, 1], [0, 0],[0,0]], "CONSTANT")
        dinputs_dy = tf.pad(dinputs_dy, [[0,0],[0, 0], [0, 1],[0,0]], "CONSTANT")
        tot_var = tf.add(tf.abs(dinputs_dx), tf.abs(dinputs_dy))
        tot_var = tf.reduce_sum(tot_var, axis =3, keep_dims=True)
        return tot_var

def TVlosses(eye_mask, ori_img, flow, lcm_map):
    with tf.variable_scope('TVlosses'):
        # eyeball_TVloss
        # calculate TV (dFlow(p)/dx  + dFlow(p)/dy)
        TV_flow = TVloss(flow)
        # calculate the (1-D(p))
        img_gray = tf.reduce_mean(ori_img, axis = 3, keep_dims=True)
        ones = tf.ones(shape = tf.shape(img_gray))
        bright = ones - img_gray
        # calculate the F_e(p)
        eye_mask = tf.expand_dims(eye_mask, axis = 3)   
        weights = tf.multiply(bright,eye_mask)  
        TV_eye = tf.multiply(weights,TV_flow)

        # eyelid_TVloss
        lid_mask = ones - eye_mask
        TV_lid = tf.multiply(lid_mask,TV_flow) 
        
        TV_eye = tf.reduce_sum(TV_eye, axis = [1,2,3])
        TV_lid = tf.reduce_sum(TV_lid, axis = [1,2,3])

        # lcm_map loss
        dist2cent = center_weight(tf.shape(lcm_map), base=0.005, boundary_penalty=3.0)
        TV_lcm = dist2cent*TVloss(lcm_map)
        TV_lcm = tf.reduce_sum(TV_lcm, axis = [1,2,3])
        
        return tf.reduce_mean(TV_eye, axis=0), tf.reduce_mean(TV_lid, axis=0), tf.reduce_mean(TV_lcm, axis=0)
    
def center_weight(shape, base=0.005, boundary_penalty=3.0):
    with tf.variable_scope('center_weight'):
        temp = boundary_penalty - base
        x = tf.pow(tf.abs(tf.lin_space(-1.0, 1.0,shape[1])),8)
        y = tf.pow(tf.abs(tf.lin_space(-1.0, 1.0,shape[2])),8)      
        X, Y = tf.meshgrid(y, x)
        X = tf.expand_dims(X, axis=2)
        Y = tf.expand_dims(Y, axis=2)
        dist2cent = temp*tf.sqrt(tf.reduce_sum(tf.square(tf.concat([X,Y], axis=2)), axis=2)) + base
        dist2cent = tf.expand_dims(tf.tile(tf.expand_dims(dist2cent, axis=0), [shape[0],1,1]), axis=3)
        return dist2cent

def lcm_adj(lcm_wgt):
    dist2cent = center_weight(tf.shape(lcm_wgt), base=0.005, boundary_penalty=3.0)
    with tf.variable_scope('lcm_adj'):
        _, loss = tf.split(lcm_wgt, [1,1], 3)
        loss = tf.reduce_sum(tf.abs(loss)*dist2cent, axis = [1,2,3])
        return tf.reduce_mean(loss, axis=0)
    
def loss(img_pred, img_, eye_mask, input_img, flow, lcm_wgt):
    with tf.variable_scope('losses'):
        loss_img = dist_loss(img_pred, img_, method = "L2")

        loss_eyeball, loss_eyelid, loss_lcm= TVlosses(eye_mask, input_img, flow, lcm_wgt)        
        loss_lcm_adj = lcm_adj(lcm_wgt)

        losses = loss_img + loss_eyeball + loss_eyelid + loss_lcm_adj + loss_lcm 
        tf.add_to_collection('losses', losses)
        return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss_img