import tensorflow as tf
import numpy as np

import tf_utils
import transformation
img_crop = 3
def cnn_blk(input_img, filters, kernel_size, phase_train, name = 'cnn_blk'):
    with tf.variable_scope(name) as scope:
        cnn = tf.layers.conv2d(inputs=input_img, filters=filters, kernel_size=kernel_size, padding="same", activation=None, use_bias=True, name="cnn")
        bn_act = tf.nn.relu(tf_utils.batch_norm(cnn, phase_train), name= "act")
        return bn_act

def create_agl_map(inputs, height, width,feature_dims):
    with tf.name_scope("create_agl_map"):
        batch_size = tf.shape(inputs)[0]
        ret = tf.reshape(tf.tile(inputs,tf.constant([1,height*width])), [batch_size,height,width,feature_dims])
        return ret

def apply_light_weight(batch_img, light_weight):
    with tf.name_scope('apply_light_weight'):
        img_wgts, pal_wgts = tf.split(light_weight, [1,1], 3)
        img_wgts = tf.tile(img_wgts, [1,1,1,3])
        pal_wgts = tf.tile(pal_wgts, [1,1,1,3])
        palette = tf.ones(tf.shape(batch_img), dtype = tf.float32)
        ret = tf.add(tf.multiply(batch_img, img_wgts), tf.multiply(palette, pal_wgts))
        return ret    
    
### define your inference model function ###
def trans_module(input_map, structures, phase_train, name="trans_module"):
    with tf.variable_scope(name) as scope:
        layer_0 = cnn_blk(input_map, structures['depth'][0], structures['filter_size'][0], phase_train, name = 'cnn_blk_0')
        layer_1 = cnn_blk(layer_0, structures['depth'][1], structures['filter_size'][1], phase_train, name = 'cnn_blk_1')
        layer_2 = cnn_blk(layer_1, structures['depth'][2], structures['filter_size'][2], phase_train, name = 'cnn_blk_2')
        layer_3 = cnn_blk(layer_2, structures['depth'][3], structures['filter_size'][3], phase_train, name = 'cnn_blk_3')
        layer_4 = tf.nn.tanh(tf.layers.conv2d(inputs=layer_3,
                                              filters=structures['depth'][4],
                                              kernel_size=structures['filter_size'][4],
                                              padding="same", activation=None, use_bias=False, name="layer_4"))
        return layer_4, layer_3
    
def lcm_module(lcm_input, structures, phase_train, name="lcm_module"):
    with tf.variable_scope(name) as scope:
        lcm_0 = cnn_blk(lcm_input, structures['depth'][0], structures['filter_size'][0], phase_train, name = 'cnn_blk_0')
        lcm_1 = cnn_blk(lcm_0, structures['depth'][1], structures['filter_size'][1], phase_train, name = 'cnn_blk_1')
        lcm_2 = tf.layers.conv2d(inputs=lcm_1, filters=structures['depth'][2], kernel_size=structures['filter_size'][2], padding="same", activation=None, use_bias=False, name='lcm_2')
        return lcm_2


# def inference(input_img, input_fp, input_ang, phase_train, eye_masks, lid_masks, conf):
def inference(input_img, input_fp, input_ang, phase_train, conf):
    corse_layer = {'depth':(16,32,32,32,2), 'filter_size':([5,5],[3,3],[3,3],[1,1],[1,1])}
    fine_layer = {'depth':(16,32,32,32,2), 'filter_size':([5,5],[3,3],[3,3],[1,1],[1,1])}
    lcm_layer = {'depth':(8,8,2), 'filter_size':([1,1],[1,1],[3,3])}
    with tf.variable_scope('DeepWarp'):
        '''agl encode module'''
        with tf.variable_scope('embed_angle'):
            agl1 =  tf.nn.relu(tf.layers.dense(inputs=input_ang, units=16, activation=None, name="encode_1"))
            agl2 =  tf.nn.relu(tf.layers.dense(inputs=agl1, units=16, activation=None, name="encode_2"))
            agl3 =  tf.nn.relu(tf.layers.dense(inputs=agl2, units=conf.encoded_agl_dim, activation=None, name="encode_3"))
            agl_map = create_agl_map(agl3, conf.height, conf.width, conf.encoded_agl_dim)
        
        input_maps = tf.concat([input_img,input_fp,agl_map],axis=3, name='input_maps')
        '''coarse module'''
        resized_input_maps = tf.layers.average_pooling2d(inputs=input_maps, pool_size=[2,2], strides=2, padding='same', name='coarse_input')       
        coarse_out, cours_f = trans_module(resized_input_maps, corse_layer, phase_train, name='coarse_module')     
        coarse_flow = tf.image.resize_images(coarse_out, (conf.height,conf.width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # coarse_flow = tf.layers.average_pooling2d(inputs=coarse_out_resize, pool_size=[2,2], strides=1, padding='same')
        coarse_img = transformation.apply_transformation(flows=coarse_flow, img=input_img, num_channels=3)

        '''fine module'''
        fine_input_maps = tf.concat([input_maps,coarse_img,coarse_flow], axis=3, name='fine_input')
        res_flow, fine_f = trans_module(fine_input_maps, fine_layer, phase_train, name='fine_module')

        '''flows'''
        flow = tf.add(coarse_flow, res_flow, name = 'D')
        cfw_img = transformation.apply_transformation(flows = flow, img = input_img, num_channels=3)

        '''lcm module'''
        coarse_features = tf.image.resize_images(cours_f, (conf.height, conf.width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        lcm_input = tf.concat([coarse_features,fine_f], axis=3, name='lcm_input')
        lcm_out = lcm_module(lcm_input, lcm_layer, phase_train, name="lcm_module")
        
        '''spatial transformation'''
        ppx_weights = tf.nn.softmax(lcm_out)
        lcm_img = apply_light_weight(batch_img=cfw_img, light_weight=ppx_weights)
        return lcm_img, flow, ppx_weights

def dist_loss(y_pred, y_, method="L2"):
    with tf.variable_scope('img_dist_loss'):
        loss = 0
        if(method == "L2"):
            loss = tf.sqrt(tf.clip_by_value(tf.reduce_sum(tf.square(y_pred - y_), axis=3, keepdims = True), 1e-10, 10))
#             loss = tf.reduce_sum(tf.square(y_pred - y_), axis=3, keepdims = True)
        elif (method == "MAE"):
            loss = tf.abs(y_pred - y_)
            
        loss = loss[:,img_crop:(-1)*img_crop,img_crop:(-1)*img_crop,:]
        loss = tf.reduce_sum(loss, axis=[1,2,3])
#         loss = 0.5*tf.reduce_mean(loss, axis=[1,2,3])
        return tf.reduce_mean(loss, axis=0)

def TVloss(inputs):
    with tf.variable_scope('TVloss'):
        dinputs_dx = inputs[:,:-1,:,:] - inputs[:,1:,:,:]
        dinputs_dy = inputs[:,:,:-1,:] - inputs[:,:,1:,:]
        dinputs_dx = tf.pad(dinputs_dx, [[0,0],[0,1],[0,0],[0,0]], "CONSTANT")
        dinputs_dy = tf.pad(dinputs_dy, [[0,0],[0,0],[0,1],[0,0]], "CONSTANT")
        tot_var = tf.add(tf.abs(dinputs_dx), tf.abs(dinputs_dy))
        tot_var = tf.reduce_sum(tot_var, axis=3, keepdims=True)
        return tot_var

def TVlosses(eye_mask, ori_img, flow, lcm_map):
    with tf.variable_scope('TVlosses'):
        # eyeball_loss
        # calculate TV (dFlow(p)/dx  + dFlow(p)/dy)
        TV_flow = TVloss(flow)
        # calculate the (1-D(p))
        img_gray = tf.reduce_mean(ori_img, axis=3, keepdims=True)
        ones = tf.ones(shape=tf.shape(img_gray))
        bright = ones - img_gray
        # calculate the F_e(p)
        eye_mask = tf.expand_dims(eye_mask, axis=3)   
        weights = tf.multiply(bright, eye_mask)  
        TV_eye = tf.multiply(weights, TV_flow)

        # eyelid_loss
        lid_mask = ones - eye_mask
        TV_lid = tf.multiply(lid_mask, TV_flow) 
        
        TV_eye = tf.reduce_sum(TV_eye, axis=[1,2,3])
        TV_lid = tf.reduce_sum(TV_lid, axis=[1,2,3])

        # lcm_map loss
        dist2cent = center_weight(tf.shape(lcm_map))
        TV_lcm = dist2cent*TVloss(lcm_map)
        TV_lcm = tf.reduce_sum(TV_lcm, axis=[1,2,3])
        
        return tf.reduce_mean(TV_eye, axis=0), tf.reduce_mean(TV_lid, axis=0), tf.reduce_mean(TV_lcm, axis=0)
    
def center_weight(shape, base=0.005, boundary_penalty=3.0):
    with tf.variable_scope('center_weight'):
        x = tf.pow(tf.abs(tf.lin_space(-1.0, 1.0, shape[1])), 8)
        y = tf.pow(tf.abs(tf.lin_space(-1.0, 1.0, shape[2])), 8)      
        X, Y = tf.meshgrid(y, x)
        X = tf.expand_dims(X, axis=2)
        Y = tf.expand_dims(Y, axis=2)
        dist2cent = boundary_penalty*tf.sqrt(tf.reduce_sum(tf.square(tf.concat([X,Y], axis=2)), axis=2)) + base
        dist2cent = tf.expand_dims(tf.tile(tf.expand_dims(dist2cent, axis=0), [shape[0],1,1]), axis=3)
        return dist2cent

def lcm_adj(lcm_wgt):
    dist2cent = center_weight(tf.shape(lcm_wgt))
    with tf.variable_scope('lcm_adj'):
        _, loss = tf.split(lcm_wgt, [1,1], 3)
        loss = tf.reduce_sum(tf.abs(loss)*dist2cent, axis=[1,2,3])
        return tf.reduce_mean(loss, axis=0)
    
def loss(img_pred, img_, eye_mask, input_img, flow, lcm_wgt, loss_combination):
    with tf.variable_scope('losses'):
        loss_img = dist_loss(img_pred, img_)

        loss_eyeball, loss_eyelid, loss_lcm = TVlosses(eye_mask, input_img, flow, lcm_wgt)        
        loss_lcm_adj = lcm_adj(lcm_wgt)
        if loss_combination == 'l2sc':
            losses = loss_img + loss_eyeball + loss_eyelid + loss_lcm_adj + loss_lcm
        elif loss_combination == 'l2s':
            losses = loss_img + loss_eyeball + loss_eyelid
        else:
            losses = loss_img 
        tf.add_to_collection('losses', losses)
        return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss_img