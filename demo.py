# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

import sys, time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse

from simple_tracker import Simple_tracker as st

"""hyper parameters"""
use_cuda = True
torch.cuda.empty_cache()
# boxes = [[[xmin/res, ymin/res, xmax/res, ymax/res, confidence, confidence, class]]]

def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    print(boxes)

    plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)


def detect_cv2_video(cfgfile, weightfile, videofile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(videofile)
    cap.set(3, 1280)
    cap.set(4, 720)
    print("Starting the YOLO loop...")

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    mst = st((1280,720), iou_thre=0.1, occlude_delay=30, target_label=0)

    frame = 0
    fps_timer = time.time()
    while True:
        frame += 1
        ret, img = cap.read()
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)

        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

        result_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

        ##################################################################
        mst.obj_tracking(boxes, frame)
        # print(mst.tracking_obj.loc[mst.tracking_obj['Frame']==frame])
        # mst.plot_tracking_onimage(frame, img)
        df = mst.tracking_obj.loc[mst.tracking_obj['Frame']==frame,:]
        for i in range(df.shape[0]):
            x0 = int((df.iloc[i,:]['xmin'] + df.iloc[i,:]['xmax']) / 2)
            y0 = int((df.iloc[i,:]['ymin'] + df.iloc[i,:]['ymax']) / 2)
            result_img = cv2.putText(result_img, str(df.iloc[i,:]['ID']), 
                                 (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255))
        ##################################################################
        result_img = cv2.putText(result_img, 'FPS:'+str(round(1/(time.time()-fps_timer),2)), 
                                 (0,25), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0))
        fps_timer = time.time()

        cv2.imshow('Yolo demo', result_img)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    cap.release()


def detect_cv2_camera(cfgfile, weightfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(2)
    if not (cap.isOpened()):
        raise Exception('Selected camera cannot be opened!')

    cap.set(3, 1280)
    cap.set(4, 720)
    print("Starting the YOLO loop...")

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    mst = st((1280,720), iou_thre=0.1, occlude_delay=30, target_label=None)

    frame = 0
    fps_timer = time.time()
    while True:
        frame += 1
        ret, img = cap.read()
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

        result_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

        ##################################################################
        mst.obj_tracking(boxes, frame)
        # print(mst.tracking_obj.loc[mst.tracking_obj['Frame']==frame])
        # mst.plot_tracking_onimage(frame, img)
        df = mst.tracking_obj.loc[mst.tracking_obj['Frame']==frame,:]
        for i in range(df.shape[0]):
            x0 = int((df.iloc[i,:]['xmin'] + df.iloc[i,:]['xmax']) / 2)
            y0 = int((df.iloc[i,:]['ymin'] + df.iloc[i,:]['ymax']) / 2)
            result_img = cv2.putText(result_img, str(df.iloc[i,:]['ID']), 
                                 (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255))
        ##################################################################
        result_img = cv2.putText(result_img, 'FPS:'+str(round(1/(time.time()-fps_timer),2)), 
                                 (0,25), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0))
        fps_timer = time.time()

        cv2.imshow('Yolo demo', result_img)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    cap.release()


def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch1.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        default='./data/mscoco2017/train2017/190109_180343_00154162.jpg',
                        help='path of your image file.', dest='imgfile')
    parser.add_argument('-vdfile', type=str,
                        default='./data/video.mp4',
                        help='path of your video file.', dest='vdfile')
    parser.add_argument('-mode', type=int,
                        default=0,
                        help='define the mode 0-img(default), 1-camera, 2-video.', dest='mode')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    if args.mode == 0:
        detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_skimage(args.cfgfile, args.weightfile, args.imgfile)
    elif args.mode == 1:
        detect_cv2_camera(args.cfgfile, args.weightfile)
    elif args.mode == 2:
        detect_cv2_video(args.cfgfile, args.weightfile, args.vdfile)
    else:
        raise Exception('Wrong mode.')
