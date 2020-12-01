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

def parse_args():
    parser = argparse.ArgumentParser(description='RFCN Demo')
    parser.add_argument('--inputdir', default='demo/ooo',help="path to input directory")#'./demo/input_imgs/'#'./demo/mytake/'#./demo/products-yahoo/
    parser.add_argument('--outputdir',default='oootest',help="path to output directory")#'./myoutput/'#'./mytaketest/'#./products-yahoo/
    parser.add_argument('--cropjson',default='bbox_info.json',help="path to output json file including bounding box coordinate information")
    parser.add_argument('--batchsize',default=1000,type=int,help="size of batch processing to speed up detection")
    parser.add_argument('--conf_thres',default=0.3,type=float,help="keep the candidates whose confidence value is higher than confidence threshold. If the number of filtered candidate is zero, we keep the at least one with the largest confidence as the detection result")
    parser.add_argument('--topk',default=1,type=int,help="obtain at most k detection results. if the number of possible results is lower than k, the number of outputs will be lower than k")
    parser.add_argument('--mode',default='crop',choices=['draw', 'crop'],type=str,help="image cropping or drawing")
    parser.add_argument('--nonms', action='store_true',help="set true to keep redundant bounding boxes (i.e. disable non-maximum suppression)")
    args = parser.parse_args()
    return args
args = parse_args()
"""
# Cropping Example
python run.py --mode crop --topk 100
python run.py --mode crop --topk 1 --nonms
python run.py --mode crop --topk 1



# Drawing Example
python run.py --mode draw --topk 2
python run.py --mode draw --topk 5
python run.py --mode draw --topk 5 --nonms
"""

# colorcode in (B,G,R) format
colorcode = {}
interval = int(255 / args.topk)
for i, v in enumerate(range(0,256,interval)):
    blue, green, red = 255-v, v, v
    colorcode[i] = (blue, green, red)


args = parse_args()

if not os.path.exists(args.outputdir):
    os.mkdir(args.outputdir)

def nms_python(bboxes,pscores,threshold):
    '''
    # Modified from https://github.com/satheeshkatipomu/nms-python/blob/master/NMS%20using%20Python%2C%20Tensorflow.ipynb
    NMS: first sort the bboxes by scores , 
        keep the bbox with highest score as reference,
        iterate through all other bboxes, 
        calculate Intersection Over Union (IOU) between reference bbox and other bbox
        if iou is greater than threshold,then discard the bbox and continue.
        
    Input:
        bboxes(numpy array of tuples) : Bounding Box Proposals in the format (x_min,y_min,x_max,y_max,confidence).
        pscores(numpy array of floats) : confidance scores for each bbox in bboxes.
        threshold(float): Overlapping threshold above which proposals will be discarded.
        
    Output:
        filtered_bboxes(numpy array) :selected bboxes for which IOU is less than threshold. 
    '''
    #Unstacking Bounding Box Coordinates
    bboxes = bboxes.astype('float')
    x_min = bboxes[:,0]
    y_min = bboxes[:,1]
    x_max = bboxes[:,2]
    y_max = bboxes[:,3]
    
    #Sorting the pscores in descending order and keeping respective indices.
    sorted_idx = pscores.argsort()[::-1]
    #Calculating areas of all bboxes.Adding 1 to the side values to avoid zero area bboxes.
    bbox_areas = (x_max-x_min+1)*(y_max-y_min+1)
    
    #list to keep filtered bboxes.
    filtered = []
    while len(sorted_idx) > 0:
        #Keeping highest pscore bbox as reference.
        rbbox_i = sorted_idx[0]
        #Appending the reference bbox index to filtered list.
        filtered.append(rbbox_i)
        
        #Calculating (xmin,ymin,xmax,ymax) coordinates of all bboxes w.r.t to reference bbox
        overlap_xmins = np.maximum(x_min[rbbox_i],x_min[sorted_idx[1:]])
        overlap_ymins = np.maximum(y_min[rbbox_i],y_min[sorted_idx[1:]])
        overlap_xmaxs = np.minimum(x_max[rbbox_i],x_max[sorted_idx[1:]])
        overlap_ymaxs = np.minimum(y_max[rbbox_i],y_max[sorted_idx[1:]])
        
        #Calculating overlap bbox widths,heights and there by areas.
        overlap_widths = np.maximum(0,(overlap_xmaxs-overlap_xmins+1))
        overlap_heights = np.maximum(0,(overlap_ymaxs-overlap_ymins+1))
        overlap_areas = overlap_widths*overlap_heights
        
        #Calculating IOUs for all bboxes except reference bbox
        ious = overlap_areas/(bbox_areas[rbbox_i]+bbox_areas[sorted_idx[1:]]-overlap_areas)
        
        #select indices for which IOU is greather than threshold
        delete_idx = np.where(ious > threshold)[0]+1
        delete_idx = np.concatenate(([0],delete_idx))
        
        #delete the above indices
        sorted_idx = np.delete(sorted_idx,delete_idx)
        
    
    #Return filtered bboxes
    return bboxes[filtered]

def batch_proc(sym,image_names_full):
    '''
    Input:
        sym(symbol objects of mxnet) : detection network
        image_names_full(list of strings) : list of string store images full path. e.g. [products-momo/xxx.jpg, products-momo/ooo.jpg]

    Output:
        output : [inputfile, (y1,x1,y2,x2,conf)] where inputfile stores input image filepath and (y1,x1,y2,x2,conf) stores mulitple object proposal coordinates and their corresponding confidence value.
    '''
    # load raw data(image)
    expand_ratio_list = []
    data = []
    im_list = []
    im_info_list = []
    for im_name in image_names_full:
        print(im_name)
        #https://stackoverflow.com/questions/50898034/how-replace-transparent-with-a-color-in-pillow
        im = PIL.Image.open(im_name)
        if im.mode == "RGB":
            im = cv2.imread(im_name, cv2.IMREAD_COLOR)
        else:
            im = im.convert("RGBA")
            new_canvas = PIL.Image.new("RGB",im.size,"WHITE")
            new_canvas.paste(im, (0,0), im)
            im = cv2.cvtColor(np.array(new_canvas), cv2.COLOR_RGB2BGR)
        #===========================================================================================
        target_size, max_size = config.TEST.SCALES[0][0], config.TEST.SCALES[0][1]
        im_height, im_width, im_channel = im.shape # (4096,2304,3)
        im, im_scale = resize(im, target_size, max_size, stride=config.network.RPN_FEAT_STRIDE)
        small_height, small_width, small_channel = im.shape
        expand_ratio_list.append([float(im_height)/float(small_height),float(im_width)/float(small_width)])
        im_list.append(im)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        im_info_list.append(im_info)
        data.append({'data': im_tensor, 'im_info': im_info}) #, 'im_ids': mx.nd.array([[1]])})
    #len(data)=29
    #data[0].keys() = ['data', 'im_info']
    #data[0]['data'].shape = (1,3,512,512)
    #data[0]['im_info'] = array([[512.   , 512.   ,   0.512]], dtype=float32)
    #---------------------
    #len(im_list)=29
    #im_list[0].shape = 512, 512, 3
    #---------------------
    #len(im_info_list)=29
    #im_info_list[0] = array([[512.   , 512.   ,   0.512]], dtype=float32)
    
    # symbol preparation
    data_names = ['data', 'im_info'] #, 'im_ids']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    #len(data)=29
    #data[0][0] = (1,3,512,512)
    #data[0][1] = [[512.    512.      0.512]] <NDArray 1x3 @cpu(0)>
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

    # prepare evaluation data
    eval_list = list(range(len(image_names_full)))
    im_info_list_eval = [im_info_list[i] for i in eval_list]
    eval_data = [data[i] for i in eval_list]
    eval_im_list = [im_list[i] for i in eval_list]
    image_names_eval = [image_names_full[i] for i in eval_list]

    # extract roi_pooled features for evaluation / visualization
    rois = []
    objectness_scores = []
    

    # get eval data based on the train val split
    count = 0
    total = len(eval_data)
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
        roi_this = bbox_pred(mod.get_outputs()[0].asnumpy().reshape((-1, 5))[:, 1:], np.array([0.1, 0.1, 0.2, 0.2]) * mod.get_outputs()[2].asnumpy()[0])
        #roi_this.shape = (300, 4)
        roi_this = clip_boxes(roi_this, im_info_list_eval[idx][0][:2])
        #roi_this.shape = (300, 4)
        roi_this = roi_this / im_info_list_eval[idx][0][2]
        #roi_this.shape = (300, 4)
        rois.append(roi_this)
        objectness_scores.append(mod.get_outputs()[1].asnumpy())
        #if count % 100 == 0: print(str(count) + '/' + str(total) + ': {:.4f} seconds spent.'.format(toc()))
    
    #objectness_scores.shape = (39, 1, 300, 2)
    output = []
    for idx in range(len(rois)):
        scale = im_info_list_eval[idx][0][2]
        pscores = objectness_scores[idx][0][:, [1]].flatten()#obtain confidence value of all proposals
        if args.nonms:
            # KEEP Redundant Bounding Boxes
            max_index_list = np.argsort(pscores)[-1*args.topk:]
            filtered_boxes_list = []
            for it, max_index in enumerate(max_index_list):
                if it == 0 or pscores[max_index] > args.conf_thres:#keep at least one candidate and remove the remaining candidates whose confidence value is lower than threshold
                    boxes = rois[idx][max_index].astype('f')
                    x1, y1, x2, y2 = boxes * scale
                    x1 *= expand_ratio_list[idx][1]
                    x2 *= expand_ratio_list[idx][1]
                    y1 *= expand_ratio_list[idx][0]
                    y2 *= expand_ratio_list[idx][0]
                    x1, y1, x2, y2, conf = int(x1), int(y1), int(x2), int(y2), float(pscores[max_index])
                    #print("bounding box info:",x1,y1,x2,y2)
                    filtered_boxes_list.append((x1,y1,x2,y2,conf))
            boxes_list = filtered_boxes_list
        else:
            # Remove Redundant Bounding Boxes with Non-Maximum Suppression (NMS)
            bboxes = []
            for p in range(len(pscores)):
                boxes = rois[idx][p].astype('f')
                x1, y1, x2, y2 = boxes *scale
                x1 *= expand_ratio_list[idx][1]
                x2 *= expand_ratio_list[idx][1]
                y1 *= expand_ratio_list[idx][0]
                y2 *= expand_ratio_list[idx][0]
                x1, y1, x2, y2, conf = int(x1), int(y1), int(x2), int(y2), float(pscores[p])#casting
                bboxes.append((x1,y1,x2,y2,conf))
            bboxes = np.array(bboxes)#dtype=float64
            bboxes_after_nms = nms_python(bboxes,pscores,0.0)#dtype=float64
            filtered_boxes_list = []
            for it, bbbb in enumerate(bboxes_after_nms):
                x1, y1, x2, y2, pscore = bbbb
                x1, y1, x2, y2, conf = int(x1), int(y1), int(x2), int(y2), float(pscore)
                if it == 0 or conf > args.conf_thres:#keep at least one candidate and remove the remaining candidates whose confidence value is lower than threshold
                    filtered_boxes_list.append((x1,y1,x2,y2,conf))#casting
            boxes_list = filtered_boxes_list
        # Filter again
        num_box = min(len(boxes_list),args.topk)
        chosen_boxes_list = boxes_list[:num_box]
        # prepare output format
        tmp = []
        for it, boxes in enumerate(chosen_boxes_list):
            x1, y1, x2, y2, conf = boxes
            tmp.append((y1,x1,y2,x2,conf))
        inputfile = image_names_full[idx]
        output.append([inputfile,tmp])
    return output

def main():
    # get symbol (a symbol can be regarded as a network in mxnet)
    pprint.pprint(config)
    sym_inst = resnet_mx_101_e2e_3k_demo()
    sym = sym_inst.get_symbol_rcnn(config, is_train=False)
    # load all images
    image_names_full = glob(os.path.join(args.inputdir,"*"))#./demo/input_imgs/*
    # prepare a dictionary to store bounding box information
    cropcoord_dict = {}
    # continue from last run
    if os.path.exists(args.cropjson):
        with open(args.cropjson,"r") as f:
            cropcoord_dict = json.load(f)
    # main loop
    for i in tqdm(range(0,len(image_names_full),args.batchsize)):
        # continue or not
        continue_flag = False
        for name in image_names_full[i:i+args.batchsize]:
            name = os.path.join(*(name.split(os.sep)[-2:]))
            if name in cropcoord_dict:
                continue_flag = True
                break
        if continue_flag: continue
        # batch processing
        batch_image_names_full = image_names_full[i:i+args.batchsize]
        output = batch_proc(sym,batch_image_names_full)
        # output
        for out in output:
            # image part
            im_name, bboxlist = out
            im = PIL.Image.open(im_name)
            if im.mode == "RGB":
                im = cv2.imread(im_name, cv2.IMREAD_COLOR)
            else:
                im = im.convert("RGBA")
                new_canvas = PIL.Image.new("RGB",im.size,"WHITE")
                new_canvas.paste(im, (0,0), im)
                im = cv2.cvtColor(np.array(new_canvas), cv2.COLOR_RGB2BGR)
            for it, boxes in enumerate(bboxlist):
                y1, x1, y2, x2, conf = boxes
                if args.mode == "crop":
                    # image cropping
                    outfile = os.path.join(args.outputdir,str(it)+"_"+os.path.basename(im_name))
                    crop_im = im[y1:y2,x1:x2]
                    cv2.imwrite(outfile,crop_im)
                else:
                    # draw boudning box
                    cv2.rectangle(im, (x1, y1), (x2, y2), colorcode[it], 5)#thickness=5
                    cv2.putText(im, "%2.1f%%"%(conf*100.0), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colorcode[it], 2)
                    if it == len(bboxlist)-1:
                        outfile = os.path.join(args.outputdir,os.path.basename(im_name))
                        cv2.imwrite(outfile,im)
            # json part
            fname = os.path.join(args.inputdir,os.path.basename(im_name))
            cropcoord_dict[fname] = bboxlist
        # output bounding box information
        with open(args.cropjson,"w") as f:json.dump(cropcoord_dict,f)
        print("output bounding box information to %s"%(args.cropjson))
    

if __name__ == '__main__':
    main()
