import init, os, argparse, sys, logging, pprint, cv2, pickle, json
sys.path.insert(0, 'lib')
from configs.faster.default_configs import config, update_config
import numpy as np
# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/configs/faster/res101_mx_3k.yml')

import mxnet as mx
from symbols import *

from bbox.bbox_transform import bbox_pred, clip_boxes
from demo.module import MutableModule
from demo.linear_classifier import train_model, classify_rois
from demo.vis_boxes import vis_boxes
from demo.image import resize, transform
from demo.load_model import load_param
from demo.tictoc import tic, toc
from demo.nms import nms
from symbols.faster.resnet_mx_101_e2e_3k_demo import resnet_mx_101_e2e_3k_demo, checkpoint_callback
from glob import glob
from pdb import set_trace as bp
from tqdm import tqdm
import PIL.Image



im = cv2.imread('/home/iis/Desktop/RFCN3000/SNIPER/demo/input_imgs/000001_1571968895.jpg')
print("before",im.shape)

target_size, max_size = config.TEST.SCALES[0][0], config.TEST.SCALES[0][1]
print("target_size:",target_size)
print("max_size:",max_size)

im, im_scale = resize(im, target_size, max_size, stride=config.network.RPN_FEAT_STRIDE)

print("after",im.shape)

cv2.imwrite("/home/iis/sdfsfsfsfs.jpg",im[129:377,283:465])
