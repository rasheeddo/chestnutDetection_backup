from ctypes import *
import math
import random
import cv2
import numpy as np
from random import randint
import sys
sys.path.insert(0, 'Class')
import time
import os
from LabDeltaRobot import *

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]




class IplROI(Structure):
    pass

class IplTileInfo(Structure):
    pass

class IplImage(Structure):
    pass

IplImage._fields_ = [
    ('nSize', c_int),
    ('ID', c_int),
    ('nChannels', c_int),
    ('alphaChannel', c_int),
    ('depth', c_int),
    ('colorModel', c_char * 4),
    ('channelSeq', c_char * 4),
    ('dataOrder', c_int),
    ('origin', c_int),
    ('align', c_int),
    ('width', c_int),
    ('height', c_int),
    ('roi', POINTER(IplROI)),
    ('maskROI', POINTER(IplImage)),
    ('imageId', c_void_p),
    ('tileInfo', POINTER(IplTileInfo)),
    ('imageSize', c_int),
    ('imageData', c_char_p),
    ('widthStep', c_int),
    ('BorderMode', c_int * 4),
    ('BorderConst', c_int * 4),
    ('imageDataOrigin', c_char_p)]


class iplimage_t(Structure):
    _fields_ = [('ob_refcnt', c_ssize_t),
                ('ob_type',  py_object),
                ('a', POINTER(IplImage)),
                ('data', py_object),
                ('offset', c_size_t)]


#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("../libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr


def detect(net, meta, image, thresh=.25, hier_thresh=.5, nms=.45):
    """if isinstance(image, bytes):
        # image is a filename
        # i.e. image = b'/darknet/data/dog.jpg'
        im = load_image(image, 0, 0)
    else:
        # image is an nparray
        # i.e. image = cv2.imread('/darknet/data/dog.jpg')
        im, image = array_to_image(image)
        rgbgr_image(im)
    """
    im, image = array_to_image(image)
    rgbgr_image(im)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh,
                             hier_thresh, None, 0, pnum)
    num = pnum[0]
    if nms: do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        a = dets[j].prob[0:meta.classes]
        if any(a):
            ai = np.array(a).nonzero()[0]
            for i in ai:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i],
                           (b.x, b.y, b.w, b.h)))

    res = sorted(res, key=lambda x: -x[1])
    if isinstance(image, bytes): free_image(im)
    free_detections(dets, num)
    return res


def coor(X1,X2,X3,P1,P2,P3,P):
	X=[0,0,0]
	X[0]=(P[0]-P1[0])/(P2[0]-P1[0])*(X2[0]-X1[0])+X1[0]-X2[0]/2.
	X[1]=(P[1]-P1[1])/(P3[1]-P1[1])*(X3[1]-X1[1])+X1[1]-X3[1]/2.

	return X


def coor_g(Xc,X1,X2,X3,P1,P2,P3,P):
	X=[0,0,0]
	X[0]=(P[0]-P1[0])/(P2[0]-P1[0])*(X2[0]-X1[0])+Xc[0]-X2[0]/2.
	X[1]=(P[1]-P1[1])/(P3[1]-P1[1])*(X3[1]-X1[1])+Xc[1]-X3[1]/2.

	return X

def map(val, in_min, in_max, out_min, out_max):

    return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def runOnVideo(net, meta, vid_source, thresh=.15, hier_thresh=.5, nms=.45):
        Xlength = 670.0
        Ylength = 500.0
        #X1=[0.   ,0.   ,0.   ]
        X2=[Xlength ,0    ,0.   ]#X2=[420. ,0    ,0.   ]
        X3=[0. ,-Ylength  ,0.   ]#X3=[0. ,-230.   ,0.   ]
        X1=[0   , 0   ,0.   ]
       # X2=[230. ,0    ,0.   ]
       # X3=[0. ,-170.   ,0.   ]
        X =[0.   ,0.   ,0.   ]
        P1=[1,1,0]
        P2=[640,0,0]
        P3=[0,480,0]
        #Xc =[200.   ,-150.   ,0.   ]
        Xc =[0.   ,0.   ,0.   ]
        deltarobot = DeltaRobot()
        deltarobot.RobotTorqueOn()
        deltarobot.GripperTorqueOn()
        deltarobot.GoHome()
        deltarobot.GripperCheck()
        #deltarobot.DeltaGoPoint(200,-150,-580)
        #deltarobot.DeltaGoPoint(0,-0,-440)

        grabHeight = -895.0
        XBucket = 0.0
        YBucket = 480.0
        ZBucket = -430.0

        XHome = 0.0
        YHome = 0.0
        ZHome = -379.648
        
        time.sleep(0.1)
        sec = 1
        video = cv2.VideoCapture(vid_source)
 #  video.set(cv2.CV_CAP_PROP_FPS,0.5)
  
        res, frame = video.read()
        classes_box_colors = [(0, 0, 255), (0, 255, 0)]  #red for palmup --> stop, green for thumbsup --> go
        classes_font_colors = [(255, 255, 0), (0, 255, 255)]

        while video.isOpened():
            #time.sleep(0.5)
            #a=time.time()
            for i in range(6):
                res, frame = video.read()
                cv2.imshow("chest nut detection",frame)
            #print("time cam",time.time()-a)
            if not res:
                break
            a=time.time()
            
            r = detect(net, meta,   frame,thresh=.15, hier_thresh=.5)
            cv2.line(frame,(0,240),(640,240),(0, 255, 0),2) # draw center horizontal line
            cv2.line(frame,(320,0),(320,480),(0, 255, 0),2) # draw center vertical line
            #cv2.namedWindow("chest nut detection",cv2.WND_PROP_FULLSCREEN)
            #cv2.setWindowProperty("chest nut detection",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.imshow("chest nut detection",frame)
            #print("r",r)
            #print("len(r)",len(r))
            #print("time",a-time.time())
            if len(r)>0:
                #r[0] is first detected nut, r[1] is second detected nut, r[2] is third, r[3] is fourth, and so on...
                print("Detects  " + str(len(r)) + "  nuts")
                i = 0
                #for i in range(len(r)):
                cv2.rectangle(frame, (int(r[i][2][0]-r[i][2][2]/2), int(r[i][2][1]-r[i][2][3]/2)), (int(r[i][2][0]+r[i][2][2]/2), int(r[i][2][1]+r[i][2][3]/2)), (255,0,0), 2)
                cv2.namedWindow("chest nut detection",cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("chest nut detection",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                cv2.imshow("chest nut detection",frame)

                #time.sleep(0.3)
                P=[r[i][2][0],r[i][2][1]]
                #print("P",P)
                #X=coor(X1,X2,X3,P1,P2,P3,P)
                #X=coor_g(Xc,X1,X2,X3,P1,P2,P3,P)
                #print("X",X)                
                XX = map(P[0],0,640,-Xlength/2.0, Xlength/2.0) # use map() give similar result as coor_g()
                YY = map(P[1],0,480,Ylength/2.0,-Ylength/2.0)
                
                XX = -XX-0.0    # robot is placed opposite with the camera
                YY = -YY-10.0  # there is an error on Y distance
                print("XX",XX)
                print("YY",YY)

                deltarobot.GotoPoint(XX,YY,grabHeight)
                deltarobot.GripperClose()
                time.sleep(0.2)
                deltarobot.GotoPoint(XX,YY,(ZBucket-100.0))
                deltarobot.GotoPoint(XBucket,YBucket,ZBucket)
                deltarobot.GripperOpen()
                time.sleep(0.2)
                deltarobot.GotoPoint(XHome,YHome,ZBucket)

        
            if cv2.waitKey(1) == ord('q'):
                break


        video.release()





if __name__ == "__main__":
#    net = load_net("yolov2-tiny.cfg", "yolov2-tiny.weights", 0)
#    meta = load_meta("voc.data")
    net = load_net(b"../cfg/yolov3-tiny-obj_small_test.cfg", b"../backup/yolov3-tiny-obj_small_55000.weights", 0)
    meta = load_meta(b"../data/obj.data")
    vid_source = 0
    runOnVideo(net, meta, vid_source)
