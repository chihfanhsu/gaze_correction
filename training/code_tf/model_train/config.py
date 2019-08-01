#-*- coding: utf-8 -*-
import argparse

model_config = argparse.ArgumentParser()

# model
model_config.add_argument('--height', type=eval, default=48, help='')
model_config.add_argument('--width', type=eval, default=64, help='')
model_config.add_argument('--channel', type=eval, default=3, help='')
model_config.add_argument('--agl_dim', type=eval, default=2, help='')
model_config.add_argument('--encoded_agl_dim', type=eval, default=16, help='')
model_config.add_argument('--early_stop', type=eval, default=16, help='')

# hyper parameter
model_config.add_argument('--lr', type=eval, default=0.001, help='')
model_config.add_argument('--epochs', type=eval, default=500, help='')
model_config.add_argument('--batch_size', type=eval, default=256, help='')

model_config.add_argument('--dataset', type=str, default='dirl_48x64_example', help='')

# training parameter
model_config.add_argument('--tar_model', type=str, default='flx', help='')
# model_config.add_argument('--tar_model', type=str, default='deepwarp', help='')
model_config.add_argument('--loss_combination', type=str, default='l2sc', help='')
model_config.add_argument('--ef_dim', type=eval, default=12, help='')
model_config.add_argument('--eye', type=str, default="L", help='')
#load trained weight
model_config.add_argument('--load_weights', type=bool, default=False, help='')
model_config.add_argument('--easy_mode', type=bool, default=True, help='')
# model_config.add_argument('--load_weights', type=bool, default=True, help='')
# model_config.add_argument('--easy_mode', type=bool, default=False, help='')

# folders' path
model_config.add_argument('--tb_dir', type=str, default='TFboard/', help='')
model_config.add_argument('--data_dir', type=str, default='../../dataset/', help='')
model_config.add_argument('--train_dir', type=str, default='training_inputs/', help='')
model_config.add_argument('--valid_dir', type=str, default='valid_inputs/', help='')
model_config.add_argument('--weight_dir', type=str, default='pt_ckpt/', help='')

def get_config():
    config, unparsed = model_config.parse_known_args()
    print(config)
    return config, unparsed
