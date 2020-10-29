import init
import os
import argparse
import sys
import logging
import pprint
import cv2
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
import pickle
from symbols.faster.resnet_mx_101_e2e_3k_demo import resnet_mx_101_e2e_3k_demo, checkpoint_callback

def parse_args():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--thresh', help='Output threshold', default=0.5)
    parser.add_argument('--inputdir', default='./demo/input_imgs/')
    parser.add_argument('--outputdir',default='./myoutput/')
    args = parser.parse_args()
    return args

args = parse_args()

def process_mul_scores(scores, xcls_scores):
    """
    Do multiplication of objectness score and classification score to obtain the final detection score.
    """
    final_scores = np.zeros((scores.shape[0], xcls_scores.shape[1]+1))
    final_scores[:, 1:] = xcls_scores[:, :] * scores[:, [1]]
    return final_scores

def main():
    # get symbol
    pprint.pprint(config)
    sym_inst = resnet_mx_101_e2e_3k_demo()
    sym = sym_inst.get_symbol_rcnn(config, is_train=False)

    # load data
    from os import listdir
    from os.path import isfile, isdir, join
    """
    dir_names = [f for f in listdir(args.inputdir) if isdir(join(args.inputdir, f))]
    image_names_full = []
    image_num_per_class = []
    for folder in dir_names:
        image_names = [folder + '/' + a for a in listdir(args.inputdir + folder) if isfile(join(args.inputdir, folder, a)) and a.split('.')[-1] == 'jpg']
        image_names_full = image_names_full + image_names
        #image_num_per_class.append(len(image_names))
    """
    

    from glob import glob
    image_names_full = glob('./demo/input_imgs/*')

    data = []
    im_list = []
    im_info_list = []

    # load raw data(image)
    for im_name in image_names_full:
        #assert os.path.exists(cur_path + args.inputdir + im_name), ('%s does not exist'.format('./extract/' + im_name))
        #im = cv2.imread(cur_path + args.inputdir[1:] + im_name, cv2.IMREAD_COLOR | 128)
        im = cv2.imread(im_name, cv2.IMREAD_COLOR | 128)
        
        target_size = config.TEST.SCALES[0][0]
        max_size = config.TEST.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.RPN_FEAT_STRIDE)
        im_list.append(im)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        im_info_list.append(im_info)
        data.append({'data': im_tensor, 'im_info': im_info}) #, 'im_ids': mx.nd.array([[1]])})

    # symbol preparation
    data_names = ['data', 'im_info'] #, 'im_ids']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.TEST.SCALES]), max([v[1] for v in config.TEST.SCALES])))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]
    output_path = './output/chips_resnet101_3k/res101_mx_3k/fall11_whole/'
    model_prefix = os.path.join(output_path, 'CRCNN')
    arg_params, aux_params = load_param(model_prefix, config.TEST.TEST_EPOCH,convert=True, process=True)
    
    # set model
    mod = MutableModule(sym, data_names, label_names, context=[mx.gpu(0)], max_data_shapes=max_data_shape)
    mod.bind(provide_data, provide_label, for_training=False)
    mod.init_params(arg_params=arg_params, aux_params=aux_params)    

    """
    # extract feature extraction
    features = []
    indices = []
    count = 0
    total = len(image_names_full)
    print("Extracting features of %d images..." %(total))
    tic()

    for idx, im_name in enumerate(image_names_full):
        count += 1
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                        provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                        provide_label=[None])
        mod.forward(data_batch)
        out = mod.get_outputs()[4].asnumpy()
        pooled_feat = mx.ndarray.Pooling(data=mx.ndarray.array(out), pool_type='avg', global_pool=True, kernel=(7, 7))
        features.append(pooled_feat.reshape((1, -1)).asnumpy())
        indices.append(im_name)

        if count % 100 == 0:
            print(str(count) + '/' + str(total) + ': {:.4f} seconds spent.'.format(toc()))

    # train the linear classifier, get trained classifier and eval_list
    class_names = ()
    for one in dir_names:
        class_names = class_names + (one,)
    linear_classifier, eval_list = train_model(class_names, image_num_per_class, batch_size=100, learning_rate=0.001, momentum=0.9, num_epoch=250)
    eval_data = [data[i] for i in eval_list]
    eval_im_list = [im_list[i] for i in eval_list]
    im_info_list_eval = [im_info_list[i] for i in eval_list]
    image_names_eval = [image_names_full[i] for i in eval_list]
    """
    import pdb
    #pdb.set_trace()

    #len(data) = 1332
    #len(data[0]) = 2
    #data[0][0].shape = (1,3,512,512)
    #data[0][1].shape = (1,3)
    #data[1][0].shape = (1,3,384,512)
    #data[1][1].shape = (1,3)
    #eval_list = [0, ..., 1139]

    eval_list = list(range(len(image_names_full)))
    im_info_list_eval = [im_info_list[i] for i in eval_list]
    eval_data = [data[i] for i in eval_list]
    eval_im_list = [im_list[i] for i in eval_list]
    image_names_eval = [image_names_full[i] for i in eval_list]

    #pdb.set_trace()

    # extract roi_pooled features for evaluation / visualization
    rois = []
    objectness_scores = []
    roipooled_features = []

    # get eval data based on the train val split
    count = 0
    total = len(eval_data)
    print("Extracting roipooled features of %d images..." %(total))
    tic()
    for idx in range(len(eval_data)):
        count += 1
        data_batch = mx.io.DataBatch(data=[eval_data[idx]], label=[], pad=0, index=idx,
                                        provide_data=[[(k, v.shape) for k, v in zip(data_names, eval_data[idx])]],
                                        provide_label=[None])
        mod.forward(data_batch)
        #can edit the weights and put the classifier in a fully convolutional way as well....change the network, for little bit speedup!
        roipooled_conv5_feat = mx.ndarray.ROIPooling(data=mod.get_outputs()[4], rois=mod.get_outputs()[0],
                                                            pooled_size=(7, 7), spatial_scale=0.0625)
        pooled_feat = mx.ndarray.Pooling(data=roipooled_conv5_feat, pool_type='avg', global_pool=True, kernel=(7, 7))

        
        #pooled_feat.shape = (300L, 3072L, 1L, 1L)
        roipooled_features.append(pooled_feat.reshape((pooled_feat.shape[0], -1)).asnumpy())
        #roipooled_features[0].shape = (300, 3072)
        roi_this = bbox_pred(mod.get_outputs()[0].asnumpy().reshape((-1, 5))[:, 1:], np.array([0.1, 0.1, 0.2, 0.2]) * mod.get_outputs()[2].asnumpy()[0])
        #roi_this.shape = (300, 4)


        
        roi_this = clip_boxes(roi_this, im_info_list_eval[idx][0][:2])
        #roi_this.shape = (300, 4)
        
        roi_this = roi_this / im_info_list_eval[idx][0][2]
        #roi_this.shape = (300, 4)
        rois.append(roi_this)
        objectness_scores.append(mod.get_outputs()[1].asnumpy())

        if count % 100 == 0: print(str(count) + '/' + str(total) + ': {:.4f} seconds spent.'.format(toc()))
    
    #pdb.set_trace()
    #objectness_scores.shape = (39, 1, 300, 2)

    for idx in range(len(rois)):
        im_name = image_names_eval[idx]
        max_index = np.argmax(objectness_scores[idx][0][:, [1]])
        boxes = rois[idx][max_index].astype('f')

        scale = im_info_list_eval[idx][0][2]
        
        x1, y1, x2, y2 = boxes * scale
        print('testing {}'.format(im_name))
        # draw boudning box
        im = eval_im_list[idx]
        print("image size:",im.shape)
        print("bounding box info:",x1,y1,x2,y2)
        cv2.rectangle(im, (x1, y1), (x2, y2), (255,0,0), 2)

        # save
        if not os.path.exists(args.outputdir):
            os.mkdir(args.outputdir)
        outfile = args.outputdir+str(idx)+".jpg"
        print("save to {}".format(outfile))
        cv2.imwrite(outfile,im)

        




    """
    # classify the rois
    rois_cls = classify_rois(linear_classifier, roipooled_features)
    for idx in range(len(rois)):
        im_name = image_names_eval[idx]
        xcls_scores = process_mul_scores(objectness_scores[idx][0], rois_cls[idx])
        boxes = rois[idx].astype('f')
        xcls_scores = xcls_scores.astype('f')
        dets_nms = []
        for j in range(1, xcls_scores.shape[1]):
            cls_scores = xcls_scores[:, j, np.newaxis]
            cls_boxes = boxes[:, 0:4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            keep = nms(cls_dets, 0.45)
            cls_dets = cls_dets[keep, :]
            cls_dets = cls_dets[cls_dets[:, -1] > float(args.thresh), :]
            dets_nms.append(cls_dets)

        print 'testing {}'.format(im_name)
        # visualize
        im = cv2.cvtColor(eval_im_list[idx].astype(np.uint8), cv2.COLOR_BGR2RGB)
        vis_boxes(im_name, im, dets_nms, im_info_list_eval[idx][0][2], config, args.thresh, dir_names)
    """

if __name__ == '__main__':
    main()
