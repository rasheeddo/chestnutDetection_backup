from ctypes import *
import math
import random
import cv2
import numpy as np
from random import randint
import sys
sys.path.insert(0, 'Class')
from LabDeltaRobot import *
import time
import os

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


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
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


def runOnVideo(net, meta, vid_source, thresh=.5, hier_thresh=.5, nms=.45):
    ### Center of each basis. (8 bases here)
    Xc=[[ 200,-150,-580.],\
        [ -12,-150,-580.],\
        [-195,-160,-580.],\
        [-116,  -1,-580.],\
        [ 130, -14,-580.],\
        [ 200, 150,-580.],\
        [   1, 140,-580.],\
        [-180, 150,-580.]]
    ic_count=len(Xc)

	### Coordiantes defining the center of bases
    ### x lenght is 230 mm
    ### y lenght is 420 mm
    ### z = 0
    #Xc =[200.   ,-150.   ,0.   ]
    #Xc =[0.   ,0.   ,0.   ]
    X =[  0.,   0.,0.]
    X1=[  0.,   0.,0.]
    #X2=[420.,   0.,0.]
    #X3=[0.  , 230.,0.]
    X2=[230. ,0    ,0.   ]
    X3=[0. ,-170.   ,0.   ]

    ### Pixel base
    ### height = 480
    ### width = 640
    P1=[1,1,0]
    P2=[640,0,0]
    P3=[0,480,0]

    ### Delta robot initialization
    #deltarobot = DeltaRobot()
    #deltarobot.TorqueOn()
    #deltarobot.DeltaGoHome()
    #deltarobot.GripperCheck()
    #deltarobot.DeltaGoPoint(200,-150,-580)


    ### Initialization for camera
    video = cv2.VideoCapture(vid_source)
    #video.set(cv2.CV_CAP_PROP_FPS,0.5)  
    res, frame = video.read()

    ### Parameter for opencv draw object box
    classes_box_colors = [(0, 0, 255), (0, 255, 0)] 
    classes_font_colors = [(255, 255, 0), (0, 255, 255)]
    ic=0
    check = False
    test_time=time.time()
    while video.isOpened():
        #deltarobot.DeltaGoPoint(Xc[ic][0],Xc[ic][1],Xc[ic][2])
        ### Pause time for stabilization of the robot
        ### before taking picture
        #time.sleep(0.01)
        
        ### Get the last camera frame from the buffer
        ### take less than 0.2 second
        for i in range(6):
            res, frame = video.read()
            #cv2.imshow("chest nut detection",frame)

            if not res:
                print("Error with camera")
                break

            a=time.time()

            ### Detect object and pixel coordinate in local
            ### base (camera: x:1 to 640, y: 1 to 480)
            ### take between 0.1 to 0.2 second
            r = detect(net, meta,   frame,thresh=.85, hier_thresh=.5)
            #print("time",a-time.time())

            ### Check if object detected
            #print("object detected", len(r))
            if len(r)>0:
		### Object center in Pixel
                check=True
		for i in range(len(r)):
            P=[r[i][2][0],r[i][2][1]]

                
                    #print("P",P)

            ### Plot image to screen
            cv2.rectangle(frame, (int(r[0][2][0]-r[0][2][2]/2), int(r[0][2][1]-r[0][2][3]/2)), (int(r[0][2][0]+r[0][2][2]/2), int(r[0][2][1]+r[0][2][3]/2)), (255,0,0), 2)
            cv2.imshow("chest nut detection",frame)
            time.sleep(0.3)

		    ### Determine coordinates in mm
                    #X=coor(X1,X2,X3,P1,P2,P3,P)

            X = coor_g(Xc[ic],X1,X2,X3,P1,P2,P3,P)
            print("X", X)
                    #if(not deltarobot.XYZOutRange(X[0],X[1],-600)): 
                ### Grab Nut
            check=False
                	#deltarobot.DeltaGoPoint(X[0],X[1],-580)
                	#deltarobot.DeltaGoPoint(X[0],X[1],-705)
                	#deltarobot.GripperClose()
                	#time.sleep(0.25)
                	#deltarobot.DeltaGoPoint(X[0],X[1],-450)
 
                ### Put into Bucket:
                        #if(X[1]>=0):
                	  #deltarobot.DeltaGoPoint(0.,280,-400)
                	#else:
                	  #deltarobot.DeltaGoPoint(0.,-290,-400)
                	#time.sleep(0.1)
                	#deltarobot.GripperOpen()
                	#time.sleep(0.1)
                	#deltarobot.DeltaGoPoint(Xc[ic][0],Xc[ic][1],-450)
                #deltarobot.DeltaGoHome()
                #deltarobot.DeltaGoPoint(200,-150,-580)
            print("time:",time.time()-test_time)

		if(check):
		  ic=ic+1
		  if(ic>=ic_count):
            ic=0
            test_time=time.time()-test_time
            print("test time end:",test_time)
            break
		     
            ### if there is no nuts in the area, me move to next one
        elif(ic< ic_count-1):
	        ic=ic+1
 
	    ### Start again in the first one
        else:
            ic=0
            test_time=time.time()-test_time
            print("test time:",test_time)
            break 

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
