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
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--inputdir', default='./demo/products-yahoo/')#'./demo/input_imgs/'
    parser.add_argument('--outputdir',default='./products-yahoo/')#'./myoutput/'
    parser.add_argument('--batchsize',default=1000,type=int)
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

def batch_proc(sym,image_names_full):
    # load raw data(image)
    data = []
    im_list = []
    im_info_list = []
    for im_name in image_names_full:
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
        im, im_scale = resize(im, target_size, max_size, stride=config.network.RPN_FEAT_STRIDE)
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
    roipooled_features = []

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
        #if count % 100 == 0: print(str(count) + '/' + str(total) + ': {:.4f} seconds spent.'.format(toc()))
    
    #objectness_scores.shape = (39, 1, 300, 2)
    output = []
    for idx in range(len(rois)):
        im_name = image_names_eval[idx]
        max_index = np.argmax(objectness_scores[idx][0][:, [1]])
        boxes = rois[idx][max_index].astype('f')
        scale = im_info_list_eval[idx][0][2]
        x1, y1, x2, y2 = boxes * scale
        #print('testing {}'.format(im_name))
        im = eval_im_list[idx]

        # draw boudning box
        #print("image size:",im.shape)
        #print("bounding box info:",x1,y1,x2,y2)
        #cv2.rectangle(im, (x1, y1), (x2, y2), (255,0,0), 2)

        # image cropping
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #print(x1,x2,y1,y2)#(119, 423, 173, 511)
        im = im[y1:y2,x1:x2]

        # save
        if not os.path.exists(args.outputdir):
            os.mkdir(args.outputdir)
        outfile = args.outputdir+os.path.basename(image_names_full[idx])
        #print("save to {}".format(outfile))
        #cv2.imwrite(outfile,im)
        output.append([outfile, im, (y1,x1,y2,x2)])
    return output

def main():
    store = args.inputdir.split("/")[-2]#products-yahoo
    # get symbol
    pprint.pprint(config)
    sym_inst = resnet_mx_101_e2e_3k_demo()
    sym = sym_inst.get_symbol_rcnn(config, is_train=False)
    # load data
    image_names_full = glob(args.inputdir+"*")#e.g. './demo/input_imgs/*'
    # batch processing
    cropcoord_dict = {}
    if os.path.exists(store+".json"):
        with open(store+".json","r") as f:
            cropcoord_dict = json.load(f)
    for i in tqdm(range(0,len(image_names_full),args.batchsize)):
        # continue or not
        continue_flag = False
        for name in image_names_full[i:i+args.batchsize]:
            name = os.path.join(*(name.split(os.sep)[-2:]))
            if name in cropcoord_dict:
                continue_flag = True
                break
        if continue_flag: continue
        # process
        batch_image_names_full = image_names_full[i:i+args.batchsize]
        output = batch_proc(sym,batch_image_names_full)
        for (outfile, im, (y1,x1,y2,x2)) in output:
            cv2.imwrite(outfile,im)
            fname = os.path.join(store,os.path.basename(outfile))
            cropcoord_dict[fname] = [y1,x1,y2,x2]
        with open(store+".json","w") as f:
            json.dump(cropcoord_dict,f)
    

if __name__ == '__main__':
    main()
