# __author__ = 'ChienHung Chen in Academia Sinica IIS'


import os, sys
from PIL import Image, ImageTk
import PIL.Image
import json
import cv2
import pickle
import numpy as np
import matplotlib
try:
    print("[PYTHON3]")
    from tkinter import ttk
    from tkinter import *
except:
    print("[PYTHON2]")
    #from Tkinter import ttk
    from Tkinter import *
    import ttk
import xml.etree.ElementTree as ET
import argparse
import platform
import itertools
import pdb

matplotlib.use("TkAgg")

# ==========================================================

parser = argparse.ArgumentParser(description="DetVisGUI")

# dataset information
# parser.add_argument('-f', '--format', default='COCO', help='VOC or COCO dataset format')

parser.add_argument('--det_file', default=True, help='detection result file path')

# ------------ put your image paths and pickle filepaths here ------------
parser.add_argument('--score_pkl', default='./demo/cache/ob_scores.pkl')
parser.add_argument('--rois_pkl', default='./demo/cache/eval_rois.pkl')
parser.add_argument('--img_root', default='./demo/image')
parser.add_argument('--txt', default='./demo/cache/eval_index.txt')
# ------------------------------------------------------------------

parser.add_argument('-r', action='store_true', help='detection result format is (cls, img) or (img, cls)')
parser.add_argument('--det_box_color', default=(255, 255, 0), help='detection box color')
parser.add_argument('--gt_box_color', default=(255, 255, 255), help='groundtruth box color')

parser.add_argument('--output', default='output', help='image save folder')

args = parser.parse_args()

# ==========================================================

class product_dataset:
    def __init__(self):
        self.dataset = 'product'
        # ------------------------------------
        self.det_file = args.det_file
        self.data_root = args.img_root
        self.score_pkl = args.score_pkl
        self.rois_pkl = args.rois_pkl
        self.txt = args.txt
        # ------------------------------------
        self.has_anno = False
        self.mask = False

        # according txt to get image list
        self.img_list = self.get_img_list()

        self.results = self.get_det_results() if self.det_file else None
        self.aug_category = aug_category(['FG'])

        if self.det_file:
            self.img_det = {self.img_list[i]: self.results[:, i] for i in range(len(self.img_list))}

    def get_img_list(self):
        with open(self.txt, 'r') as f:
            data = f.readlines()

        return [x.strip() for x in data]


    def get_det_results(self):
        det_file = self.det_file
        
        if det_file == '':
            return None
	
        def get_pkl(name):
            try:
                with open(name, 'rb') as f: 
                    data = pickle.load(f, encoding="bytes")
            except:
                import cPickle as ccpickle
                with open(name, 'rb') as f:
                    data = ccpickle.load(f)
            return data 
        scores = get_pkl(self.score_pkl)
        rois  = get_pkl(self.rois_pkl)
        # pdb.set_trace()        

        det_results = []

        for i in range(len(scores)):
            idx = np.argsort(-1 * scores[0][0][:, 1])
            det_results.append(np.hstack([rois[i][idx], scores[i][0, idx, 1:2]])) 
        det_results = np.asarray(det_results)
        det_results = det_results[np.newaxis, :]
        return det_results

    def get_img_by_name(self, name):
        img = PIL.Image.open(os.path.join(self.data_root, name)).convert('RGB')
        return img

    def get_img_by_index(self, idx):
        img = PIL.Image.open(os.path.join(self.data_root, self.img_list[idx]))
        img = img.convert('RGB')
        return img

    def get_singleImg_gt(self, name):       # get annotations by image name
        pass

    def get_singleImg_dets(self, name):
        return self.img_det[name]

# main GUI
class vis_tool:
    def __init__(self):
        self.window = Tk()
        self.menubar = Menu(self.window)

        self.info = StringVar()
        self.info_label = Label(self.window, bg='yellow', width=4, textvariable=self.info)

        self.listBox1 = Listbox(self.window, width=50, height=25, font=('Times New Roman', 10))
        self.listBox2 = Listbox(self.window, width=50, height=12, font=('Times New Roman', 10))

        self.scrollbar1 = Scrollbar(self.window, width=15, orient="vertical")
        self.scrollbar2 = Scrollbar(self.window, width=15, orient="vertical")

        self.listBox1_info = StringVar()
        self.listBox1_label = Label(self.window, font=('Arial', 11), bg='yellow', width=4, height=1, textvariable=self.listBox1_info)

        self.listBox2_info = StringVar()
        self.listBox2_label = Label(self.window, font=('Arial', 11), bg='yellow', width=4, height=1, textvariable=self.listBox2_info)
        self.listBox2_label2 = Label(self.window, font=('Arial', 11), bg='yellow', width=4, height=1, text="Object Class : Score (IoU)")

        self.data_info = product_dataset()

        self.info.set("DATASET: {}".format(self.data_info.dataset))

        # load image and show it on the window
        self.img = self.data_info.get_img_by_index(0)
        self.photo = ImageTk.PhotoImage(self.img)
        self.label_img = Label(self.window, image=self.photo)

        self.show_det_txt = IntVar(value=1)
        self.checkbn_det_txt = Checkbutton(self.window, text='Text', font=('Arial', 10, 'bold'), variable=self.show_det_txt, command=self.change_img, fg='#0000FF')

        self.show_dets = IntVar(value=1)
        self.checkbn_det = Checkbutton(self.window, text='Detections', font=('Arial', 10, 'bold'), variable=self.show_dets, command=self.change_img, fg='#0000FF')

        self.show_gt_txt = IntVar(value=1)
        self.checkbn_gt_txt = Checkbutton(self.window, text='Text', font=('Arial', 10, 'bold'), variable=self.show_gt_txt, command=self.change_img, fg='#FF8C00')

        self.show_gts = IntVar(value=1)
        self.checkbn_gt = Checkbutton(self.window, text='Groundtruth', font=('Arial', 10, 'bold'), variable=self.show_gts, command=self.change_img, fg='#FF8C00')

        self.combo_label = Label(self.window, bg='yellow', width=10, height=1, text='Show Category', font=('Arial', 11))
        try:
            self.combo_category = ttk.Combobox(self.window, font=('Arial', 11), values=self.data_info.aug_category.combo_list)
        except:
            self.combo_category = Combobox(self.window, font=('Arial', 11), values=self.data_info.aug_category.combo_list)
        self.combo_category.current(0)

        self.th_label = Label(self.window, font=('Arial', 11), bg='yellow', width=10, height=1, text="Score Threshold")
        self.threshold = np.float32(0.5)        # because np.float32(0.7) >= float(0.7) -> False
        self.th_entry = Entry(self.window, font=('Arial', 11), textvariable=StringVar(self.window, value=str(self.threshold)), width=10)
        self.th_button = Button(self.window, text='Enter', height=1, command=self.change_threshold)

        self.iou_th_label = Label(self.window, font=('Arial', 11), bg='yellow', width=10, height=1, text="IoU Threshold")
        self.iou_threshold = np.float32(0.5)        # because np.float32(0.7) >= float(0.7) -> False
        self.iou_th_entry = Entry(self.window, font=('Arial', 11), textvariable=StringVar(self.window, value=str(self.threshold)), width=10)
        self.iou_th_button = Button(self.window, text='Enter', height=1, command=self.change_iou_threshold)

        self.find_label = Label(self.window, font=('Arial', 11), bg='yellow', width=10, height=1, text="find")
        self.find_name = ""
        self.find_entry = Entry(self.window, font=('Arial', 11), textvariable=StringVar(self.window, value=str(self.find_name)), width=10)
        self.find_button = Button(self.window, text='Enter', height=1, command=self.findname)

        self.listBox1_idx = 0  # image listBox

        # ====== ohter attribute ======
        self.img_name = ''
        self.show_img = None
        self.output = args.output

        if not os.path.isdir(self.output):
            os.makedirs(self.output)

        self.img_list = self.data_info.img_list

        # flag for find/threshold button switch focused element
        self.button_clicked = False


    def change_threshold(self, event=None):

        try:
            self.threshold = np.float32(self.th_entry.get())
            self.change_img()

            # after changing threshold, focus on listBox for easy control
            if self.window.focus_get() == self.listBox2:
                self.listBox2.focus()
            else:
                self.listBox1.focus()

            self.button_clicked = True

        except ValueError:
            self.window.title("Please enter a number as score threshold.")


    def change_iou_threshold(self, event=None):

        try:
            self.iou_threshold = np.float32(self.iou_th_entry.get())
            self.change_img()

            # after changing threshold, focus on listBox for easy control
            if self.window.focus_get() == self.listBox2:
                self.listBox2.focus()
            else:
                self.listBox1.focus()

            self.button_clicked = True

        except ValueError:
            self.window.title("Please enter a number as IoU threshold.")


    # draw groundtruth
    def draw_gt_boxes(self, img, objs):

        for obj in objs:  # objs example : [['Car', 709, 317, 119, 76] , ['Car', 500, 370, 132, 23]]
            cls_name = obj[0]

            # according combobox to decide whether to plot this category
            show_category = self.data_info.aug_category.category if self.combo_category.get() == 'All' else [self.combo_category.get()]
            if cls_name not in show_category:
                continue

            box = obj[1:]
            xmin = max(box[0], 0)
            ymin = max(box[1], 0)
            xmax = min(box[0] + box[2], self.img_width)
            ymax = min(box[1] + box[3], self.img_height)

            font = cv2.FONT_HERSHEY_SIMPLEX

            if self.show_gt_txt.get():
                if ymax + 30 >= self.img_height:
                    cv2.rectangle(img, (xmin, ymin), (xmin + len(cls_name) * 10, int(ymin - 20)), (255,140,0), cv2.FILLED)
                    cv2.putText(img, cls_name, (xmin, int(ymin - 5)), font, 0.5, (255, 255, 255), 1)
                else:
                    cv2.rectangle(img, (xmin, ymax), (xmin + len(cls_name) * 10, int(ymax + 20)), (255,140,0), cv2.FILLED)
                    cv2.putText(img, cls_name, (xmin, int(ymax + 15)), font, 0.5, (255, 255, 255), 1)

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), args.gt_box_color, 1)

        return img


    def get_iou(self, det):

        iou = np.zeros_like(det)
        GT = self.data_info.get_singleImg_gt(self.img_name)
        
        for idx, cls_objs in enumerate(det):

            category = self.data_info.aug_category.category[idx]
            BBGT = []
            for t in GT:
                if not t[0] == category: continue
                BBGT.append([t[1], t[2], t[1] + t[3], t[2] + t[4]])
            BBGT = np.asarray(BBGT)
            d = [0] * len(BBGT)  # for check 1 GT map to several det

            confidence = cls_objs[:, 4]
            BB = cls_objs[:, 0:4]   # bounding box        
             
            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]

            ind_table = {i: sorted_ind[i] for i in range(len(sorted_ind))}  # for returning original order

            iou[idx] = np.zeros(len(BB))

            if len(BBGT) > 0:
                for i in range(len(BB)): 
                    bb = BB[i, :]
            
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih
    
                    # union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                           (BBGT[:, 2] - BBGT[:, 0] + 1.) * (BBGT[:, 3] - BBGT[:, 1] + 1.) - 
                            inters)

                    overlaps = inters / uni
                    ovmax = np.max(overlaps)          # max overlaps with all gt
                    jmax = np.argmax(overlaps)        
                    
                    if ovmax > self.iou_threshold:        
                        if not d[jmax]:
                            d[jmax] = 1
                        else:                        # multiple bounding boxes map to one gt
                            ovmax = -ovmax

                    iou[idx][ind_table[i]] = ovmax   # return to unsorted order
        return iou


    def draw_all_det_boxes(self, img, single_detection):

        if self.data_info.has_anno:
            self.iou = self.get_iou(single_detection)

        for idx, cls_objs in enumerate(single_detection):
            category = self.data_info.aug_category.category[idx]

            show_category = self.data_info.aug_category.category if self.combo_category.get() == 'All' else [self.combo_category.get()]
            if category not in show_category:
                continue

            for obj_idx, obj in enumerate(cls_objs):  # objs example : [496.2, 334.8, 668.4, 425.1, 0.99] -> [xmin, ymin, xmax, ymax, confidence]

                [score, box] = [round(obj[4], 2), obj[:4]]

                if score >= self.threshold:
                    box = list(map(int, list(map(round, box))))
                    xmin = max(box[0], 0)
                    ymin = max(box[1], 0)
                    xmax = min(box[2], self.img_width)
                    ymax = min(box[3], self.img_height)

                    if not self.data_info.has_anno or self.iou[idx][obj_idx] >= self.iou_threshold:
                        color = args.det_box_color
                    else: 
                        color = (255, 0, 0)

                    if self.show_det_txt.get():
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text = category + " : " + str(score)

                        if ymax + 30 >= self.img_height:
                            cv2.rectangle(img, (xmin, ymin), (xmin + len(text) * 9, int(ymin - 20)), (0, 0, 255), cv2.FILLED)
                            cv2.putText(img, text, (xmin, int(ymin - 5)), font, 0.5, (255, 255, 255), 1)
                        else:
                            cv2.rectangle(img, (xmin, ymax), (xmin + len(text) * 9, int(ymax + 20)), (0, 0, 255), cv2.FILLED)
                            cv2.putText(img, text, (xmin, int(ymax + 15)), font, 0.5, (255, 255, 255), 1)

                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

        return img

    def draw_all_det_boxes_masks(self, img, single_detection):
        img = np.require(img, requirements=['W'])
        boxes, masks = single_detection
        
        # draw segmentation masks  reference mmdetection/mmdet/models/detectors/base.py
        if self.combo_category.get() != 'All':
            show_idx = self.data_info.aug_category.category.index(self.combo_category.get())
            masks = np.asarray([masks[show_idx]])
            boxes = np.asarray([boxes[show_idx]])
            category = self.data_info.aug_category.category[show_idx]   # fixed category

        segms = list(itertools.chain(*masks))
        bboxes = np.vstack(boxes)

        inds = np.where(np.round(bboxes[:, -1], 2) >= self.threshold)[0]

        self.color_list = []
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
            self.color_list.append('#%02x%02x%02x' % tuple(color_mask[0]))

        if self.data_info.has_anno:
            boxes2, _ = single_detection
            self.iou = self.get_iou(boxes2)
            if self.combo_category.get() != 'All':
                iou = np.asarray([self.iou[show_idx]])
            else:
                iou = self.iou

        # draw bounding box
        for idx, cls_objs in enumerate(boxes):
            if self.combo_category.get() == 'All':
                category = self.data_info.aug_category.category[idx]

            for obj_idx, obj in enumerate(cls_objs):  # objs example : [496.2, 334.8, 668.4, 425.1, 0.99] -> [xmin, ymin, xmax, ymax, confidence]
                [score, box] = [round(obj[4], 2), obj[:4]]

                if score >= self.threshold:
                    box = list(map(int, list(map(round, box))))
                    xmin = max(box[0], 0)
                    ymin = max(box[1], 0)
                    xmax = min(box[2], self.img_width)
                    ymax = min(box[3], self.img_height)

                    if not self.data_info.has_anno or iou[idx][obj_idx] >= self.iou_threshold:
                        color = args.det_box_color
                    else: 
                        color = (255, 0, 0)

                    if self.show_det_txt.get():
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text = category + " : " + str(score)

                        if ymax + 30 >= self.img_height:
                            cv2.rectangle(img, (xmin, ymin), (xmin + len(text) * 9, int(ymin - 20)), (0, 0, 255), cv2.FILLED)
                            cv2.putText(img, text, (xmin, int(ymin - 5)), font, 0.5, (255, 255, 255), 1)
                        else:
                            cv2.rectangle(img, (xmin, ymax), (xmin + len(text) * 9, int(ymax + 20)), (0, 0, 255), cv2.FILLED)
                            cv2.putText(img, text, (xmin, int(ymax + 15)), font, 0.5, (255, 255, 255), 1)

                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

        return img


    def change_img(self, event=None):
        if len(self.listBox1.curselection()) != 0:
            self.listBox1_idx = self.listBox1.curselection()[0]

        self.listBox1_info.set("Image  {:6}  / {:6}".format(self.listBox1_idx + 1, self.listBox1.size()))

        name = self.listBox1.get(self.listBox1_idx)
        self.window.title("DATASET : " + self.data_info.dataset + '   ' + name)

        img = self.data_info.get_img_by_name(name)
        self.img_width, self.img_height = img.width, img.height

        img = np.asarray(img)

        self.img_name = name
        self.img = img

        if self.data_info.has_anno and self.show_gts.get():
            objs = self.data_info.get_singleImg_gt(name)
            img = self.draw_gt_boxes(img, objs)

        if self.data_info.results is not None and self.show_dets.get():
            if self.data_info.mask == False:
                dets = self.data_info.get_singleImg_dets(name)
                img = self.draw_all_det_boxes(img, dets)
            else:
                dets = self.data_info.get_singleImg_dets(name).transpose((1, 0))
                img = self.draw_all_det_boxes_masks(img, dets)

            self.clear_add_listBox2()

        self.show_img = img
        img = PIL.Image.fromarray(img)
        img = self.scale_img(img)
        self.photo = ImageTk.PhotoImage(img)
        self.label_img.config(image=self.photo)
        self.window.update_idletasks()

        if self.img_name in os.listdir(self.output):
            self.listBox1_label.config(bg='#CCFF99')
        else:
            self.listBox1_label.config(bg='yellow')

    # ===============================================================

    def draw_one_det_boxes(self, img, single_detection, selected_idx):

        idx_counter = 0
        for idx, cls_objs in enumerate(single_detection):

            category = self.data_info.aug_category.category[idx]

            show_category = self.data_info.aug_category.category if self.combo_category.get() == 'All' else [self.combo_category.get()]
            if category not in show_category:
                continue

            for obj_idx, obj in enumerate(cls_objs):  # objs example : [496.2, 334.8, 668.4, 425.1, 0.99] -> [xmin, ymin, xmax, ymax, confidence]
                [score, box] = [round(obj[4], 2), obj[:4]]

                if score >= self.threshold:
                    if idx_counter == selected_idx:
                        box = list(map(int, list(map(round, box))))
                        xmin = max(box[0], 0)
                        ymin = max(box[1], 0)
                        xmax = min(box[2], self.img_width)
                        ymax = min(box[3], self.img_height)

                        if not self.data_info.has_anno or self.iou[idx][obj_idx] >= self.iou_threshold:
                            color = args.det_box_color
                        else: 
                            color = (255, 0, 0)

                        if self.show_det_txt.get():
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            text = category + " : " + str(score)

                            if ymax + 30 >= self.img_height:
                                cv2.rectangle(img, (xmin, ymin), (xmin + len(text) * 9, int(ymin - 20)), (0, 0, 255), cv2.FILLED)
                                cv2.putText(img, text, (xmin, int(ymin - 5)), font, 0.5, (255, 255, 255), 1)
                            else:
                                cv2.rectangle(img, (xmin, ymax), (xmin + len(text) * 9, int(ymax + 20)), (0, 0, 255), cv2.FILLED)
                                cv2.putText(img, text, (xmin, int(ymax + 15)), font, 0.5, (255, 255, 255), 1)

                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

                        return img
                    else:
                        idx_counter += 1


    def draw_one_det_boxes_masks(self, img, single_detection, selected_idx=-1):

        img = np.require(img, requirements=['W'])
        boxes, masks = single_detection

        # draw segmentation masks   # reference mmdetection/mmdet/models/detectors/base.py
        if self.combo_category.get() != 'All':
            show_idx = self.data_info.aug_category.category.index(self.combo_category.get())
            category = self.data_info.aug_category.category[show_idx]   # fixed category
            masks = np.asarray([masks[show_idx]])
            boxes = np.asarray([boxes[show_idx]])

        segms = list(itertools.chain(*masks))
        bboxes = np.vstack(boxes)

        inds = np.where(np.round(bboxes[:, -1], 2) >= self.threshold)[0]

        self.color_list = []
        for inds_idx, i in enumerate(inds):
            if inds_idx == selected_idx:
                color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                mask = maskUtils.decode(segms[i]).astype(np.bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
                self.color_list.append('#%02x%02x%02x' % tuple(color_mask[0]))

        if self.data_info.has_anno:
            if self.combo_category.get() != 'All':
                iou = np.asarray([self.iou[show_idx]])
            else:
                iou = self.iou
                
        # draw bounding box
        idx_counter = 0
        for idx, cls_objs in enumerate(boxes):
            if self.combo_category.get() == 'All':
                category = self.data_info.aug_category.category[idx]

            for obj_idx, obj in enumerate(cls_objs):  # objs example : [496.2, 334.8, 668.4, 425.1, 0.99] -> [xmin, ymin, xmax, ymax, confidence]
                [score, box] = [round(obj[4], 2), obj[:4]]

                if score >= self.threshold:
                    if idx_counter == selected_idx:
                        box = list(map(int, list(map(round, box))))
                        xmin = max(box[0], 0)
                        ymin = max(box[1], 0)
                        xmax = min(box[2], self.img_width)
                        ymax = min(box[3], self.img_height)

                        if not self.data_info.has_anno or iou[idx][obj_idx] >= self.iou_threshold:
                            color = args.det_box_color
                        else: 
                            color = (255, 0, 0)

                        if self.show_det_txt.get():
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            text = category + " : " + str(score)

                            if ymax + 30 >= self.img_height:
                                cv2.rectangle(img, (xmin, ymin), (xmin + len(text) * 9, int(ymin - 20)), (0, 0, 255), cv2.FILLED)
                                cv2.putText(img, text, (xmin, int(ymin - 5)), font, 0.5, (255, 255, 255), 1)
                            else:
                                cv2.rectangle(img, (xmin, ymax), (xmin + len(text) * 9, int(ymax + 20)), (0, 0, 255), cv2.FILLED)
                                cv2.putText(img, text, (xmin, int(ymax + 15)), font, 0.5, (255, 255, 255), 1)

                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

                        return img
                    else:
                        idx_counter += 1



    def change_obj(self, event=None):
        if len(self.listBox2.curselection()) == 0:
            self.listBox1.focus()
            return
        else:
            listBox2_idx = self.listBox2.curselection()[0]

        self.listBox2_info.set("Detected Object : {:4}  / {:4}".format(listBox2_idx + 1, self.listBox2.size()))

        name = self.listBox1.get(self.listBox1_idx)
        img = self.data_info.get_img_by_name(name)
        self.img_width, self.img_height = img.width, img.height
        img = np.asarray(img)
        self.img_name = name
        self.img = img

        if self.data_info.has_anno and self.show_gts.get():
            objs = self.data_info.get_singleImg_gt(name)
            img = self.draw_gt_boxes(img, objs)

        if self.data_info.results is not None and self.show_dets.get():

            if self.data_info.mask == False:
                dets = self.data_info.get_singleImg_dets(name)
                img = self.draw_one_det_boxes(img, dets, listBox2_idx)
            else:
                dets = self.data_info.get_singleImg_dets(name).transpose((1, 0))
                img = self.draw_one_det_boxes_masks(img, dets, listBox2_idx)

        self.show_img = img
        img = Image.fromarray(img)
        img = self.scale_img(img)
        self.photo = ImageTk.PhotoImage(img)
        self.label_img.config(image=self.photo)
        self.window.update_idletasks()

        if self.img_name in os.listdir(self.output):
            self.listBox1_label.config(bg='#CCFF99')
        else:
            self.listBox1_label.config(bg='yellow')

    # ===============================================================

    def scale_img(self, img):

        [s_w, s_h] = [1, 1]

        # if window size is (1920, 1080), the default max image size is (1440, 810)
        (fix_width, fix_height) = (1440, 810)

        # change image size according to window size
        if self.window.winfo_width() != 1:
            fix_width = (self.window.winfo_width() - self.listBox1.winfo_width() - self.scrollbar1.winfo_width() - 5)
            fix_height = int(float(fix_width) * 9 / 16)
            # fix_height = 700

        # handle image size is too big
        if img.width > fix_width:
            s_w = float(fix_width) / img.width
        if img.height > fix_height:
            s_h = float(fix_height) / img.height

        scale = min(s_w, s_h)

        img = img.resize((int(img.width * scale), int(img.height * scale)), PIL.Image.ANTIALIAS)
        return img

    def clear_add_listBox2(self):  # object listBox
        self.listBox2.delete(0, 'end')

        if self.data_info.mask == False:
            single_detection = self.data_info.get_singleImg_dets(self.img_list[self.listBox1_idx])
        else:
            single_detection, single_mask = self.data_info.get_singleImg_dets(self.img_list[self.listBox1_idx]).transpose((1,0))
        
        show_category = self.data_info.aug_category.category if self.combo_category.get() == 'All' else [self.combo_category.get()]
        num = 0

        for idx, cls_objs in enumerate(single_detection):
            category = self.data_info.aug_category.category[idx]
            
            if category not in show_category:
                continue

            for obj_idx, obj in enumerate(cls_objs):  # objs example : [496.2, 334.8, 668.4, 425.1, 0.99] -> [xmin, ymin, xmax, ymax, confidence]
                score = round(obj[4], 2)
                if score >= self.threshold:
                    if not self.data_info.has_anno:
                        self.listBox2.insert('end', category + " : " + str(score))
                    elif self.iou[idx][obj_idx] > self.iou_threshold:
                        s = "{:15} : {:5.3} ( {:<6.3})".format(category, score, abs(round(self.iou[idx][obj_idx], 2)))
                        self.listBox2.insert('end', s)
                        self.listBox2.itemconfig(num, fg="green")
                    else:
                        s = "{:15} : {:5.3} ( {:<6.3})".format(category, score, abs(round(self.iou[idx][obj_idx], 2)))
                        self.listBox2.insert('end', s)
                        self.listBox2.itemconfig(num, fg="red")
                    num += 1

        # Display Object Number
        self.listBox2_info.set("Detected Object : {:3}".format(num))

    def change_threshold_button(self, v):
        self.threshold += v

        if self.threshold <= 0:
            self.threshold = 0
        elif self.threshold >= 1:
            self.threshold = 1

        self.th_entry.delete(0, END)
        self.th_entry.insert(0, str(round(self.threshold, 2)))
        self.change_threshold()

    def change_iou_threshold_button(self, v):
        self.iou_threshold += v

        if self.iou_threshold <= 0:
            self.iou_threshold = 0
        elif self.iou_threshold >= 1:
            self.iou_threshold = 1

        self.iou_th_entry.delete(0, END)
        self.iou_th_entry.insert(0, str(round(self.iou_threshold, 2)))
        self.change_iou_threshold()

    def save_img(self):
        print('Save Image to ' + os.path.join(self.output, self.img_name))
        cv2.imwrite(os.path.join(self.output, self.img_name), cv2.cvtColor(self.show_img, cv2.COLOR_BGR2RGB))
        self.listBox1_label.config(bg='#CCFF99')


    def eventhandler(self, event):
        if self.window.focus_get() not in [self.find_entry, self.th_entry, self.iou_th_entry]:
            if platform.system() == 'Windows':
                state_1key = 8
                state_2key = 12
            else:  # 'Linux'
                state_1key = 16
                state_2key = 20

            # <KeyRelease event state=Control|Mod2 keysym=Left keycode=113 x=1337 y=617>
            # event.state = 20
            if event.state == state_2key and event.keysym == 'Left':
                self.change_iou_threshold_button(-0.1)
            elif event.state == state_2key and event.keysym == 'Right':
                self.change_iou_threshold_button(0.1) 

            # <KeyRelease event state=Mod2 keysym=Left keycode=113 x=1266 y=410>
            # event.state = 16
            elif event.state == state_1key and event.keysym == 'Left':
                self.change_threshold_button(-0.1)
            elif event.state == state_1key and event.keysym == 'Right':
                self.change_threshold_button(0.1)
            elif event.keysym == 'q':
                self.window.quit()
            elif event.keysym == 's':
                self.save_img()

            if self.button_clicked:
                self.button_clicked = False
            else:
                if event.keysym in ['KP_Enter', 'Return']:
                    self.listBox2.focus()
                    self.listBox2.select_set(0)
                elif event.keysym == 'Escape':
                    self.change_img()
                    self.listBox1.focus()


    def combobox_change(self, event=None):
        self.listBox1.focus()
        self.change_img()


    def clear_add_listBox1(self):
        self.listBox1.delete(0, 'end')  # delete listBox1 0 ~ end items

        # add image name to listBox1
        for item in self.img_list:
            self.listBox1.insert('end', item)

        self.listBox1.select_set(0)
        self.listBox1.focus()
        self.change_img()


    def findname(self, event=None):
        self.find_name = self.find_entry.get()
        new_list = []

        if self.find_name == '':
            new_list = self.data_info.img_list
        else:
            for img_name in self.data_info.img_list:
                if self.find_name[0] == "!":
                    if self.find_name[1:] not in img_name:
                        new_list.append(img_name)
                else:
                    if self.find_name in img_name:
                        new_list.append(img_name)

        if len(new_list) != 0:
            self.img_list = new_list
            self.clear_add_listBox1()
            self.clear_add_listBox2()
            self.button_clicked = True
        else:
            self.window.title("Can't find any image about '{}'".format(self.find_name))


    def run(self):
        self.window.title("DATASET : " + self.data_info.dataset)
        self.window.geometry('1280x800+350+100')

        # self.menubar.add_command(label='QUIT', command=self.window.quit)
        # self.window.config(menu=self.menubar)                               # display the menu
        self.scrollbar1.config(command=self.listBox1.yview)
        self.listBox1.config(yscrollcommand=self.scrollbar1.set)
        self.scrollbar2.config(command=self.listBox2.yview)
        self.listBox2.config(yscrollcommand=self.scrollbar2.set)

        layer1 = 0
        layer2 = 50

        # ======================= layer 1 =========================

        # combobox
        self.combo_label.grid(row=layer1 + 30, column=0, sticky=W + E + N + S, padx=3, pady=3, columnspan=6)
        self.combo_category.grid(row=layer1 + 30, column=6, sticky=W + E + N + S, padx=3, pady=3, columnspan=6)

        # show det
        self.checkbn_det.grid(row=layer1 + 40, column=0, sticky=N + S, padx=3, pady=3, columnspan=4)
        # show det text
        self.checkbn_det_txt.grid(row=layer1 + 40, column=4, sticky=N + S, padx=3, pady=3, columnspan=2)
        
        if self.data_info.has_anno != False:
            # show gt
            self.checkbn_gt.grid(row=layer1 + 40, column=6, sticky=N + S, padx=3, pady=3, columnspan=4)
            # show gt text
            self.checkbn_gt_txt.grid(row=layer1 + 40, column=10, sticky=N + S, padx=3, pady=3, columnspan=2)

        # ======================= layer 2 =========================

        self.listBox1_label.grid(row=layer2 + 0, column=0, sticky=N + S + E + W, columnspan=12)

        # find name
        self.find_label.grid(row=layer2 + 20, column=0, sticky=E + W, columnspan=4)
        self.find_entry.grid(row=layer2 + 20, column=4, sticky=E + W, columnspan=4)
        self.find_button.grid(row=layer2 + 20, column=8, sticky=E + W, pady=3, columnspan=4)

        self.scrollbar1.grid(row=layer2 + 30, column=11, sticky=N + S + W)
        self.label_img.grid(row=layer1 + 30, column=12, sticky=N + E, padx=3, pady=3, rowspan=110)
        self.listBox1.grid(row=layer2 + 30, column=0, sticky=N + S + E + W, pady=3, columnspan=11)

        if self.data_info.det_file != '':
            self.th_label.grid(row=layer2 + 40, column=0, sticky=E + W, columnspan=6)
            self.th_entry.grid(row=layer2 + 40, column=6, sticky=E + W, columnspan=3)
            self.th_button.grid(row=layer2 + 40, column=9, sticky=E + W, pady=0, columnspan=3)

            if self.data_info.has_anno != False:
                self.iou_th_label.grid(row=layer2 + 50, column=0, sticky=E + W, columnspan=6)
                self.iou_th_entry.grid(row=layer2 + 50, column=6, sticky=E + W, columnspan=3)
                self.iou_th_button.grid(row=layer2 + 50, column=9, sticky=E + W, pady=0, columnspan=3)

            self.listBox2_label.grid(row=layer2 + 60, column=0, sticky=E + W, pady=3, columnspan=12)

            if self.data_info.has_anno != False:
                self.listBox2_label2.grid(row=layer2 + 70, column=0, sticky=E + W, pady=2, columnspan=12)

            self.scrollbar2.grid(row=layer2 + 80, column=11, sticky=N + S + W, pady=3)
            self.listBox2.grid(row=layer2 + 80, column=0, sticky=N + S + E + W, pady=3, columnspan=11)

        self.clear_add_listBox1()
        self.listBox1.bind('<<ListboxSelect>>', self.change_img)
        self.listBox1.bind_all('<KeyRelease>', self.eventhandler)

        self.listBox2.bind('<<ListboxSelect>>', self.change_obj)

        self.th_entry.bind('<Return>', self.change_threshold)
        self.th_entry.bind('<KP_Enter>', self.change_threshold)
        self.iou_th_entry.bind('<Return>', self.change_iou_threshold)
        self.iou_th_entry.bind('<KP_Enter>', self.change_iou_threshold)
        self.find_entry.bind('<Return>', self.findname)
        self.find_entry.bind('<KP_Enter>', self.findname)

        self.combo_category.bind("<<ComboboxSelected>>", self.combobox_change)

        self.window.mainloop()


class aug_category:
    def __init__(self, categories):
        self.category = categories
        try:
            self.combo_list = categories.copy()
        except:
            self.combo_list = categories[:]
        self.combo_list.insert(0, 'All')
        self.all = True

if __name__ == "__main__":
    vis_tool().run()

