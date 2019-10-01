from ctypes import *
import math
import random
import sys
sys.path.insert(0, 'Class')
#from LabDeltaRobot import *
import cv2
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

    

#lib = CDLL("../libdarknet.so", RTLD_GLOBAL)
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

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):

    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res


def coor(X1,X2,X3,P1,P2,P3,P):
	X[0]=(P[0]-P1[0])/(P2[0]-P1[0])*(X2[0]-X1[0])+X1[0]-X2[0]/2. 
	X[1]=(P[1]-P1[1])/(P3[1]-P1[1])*(X3[1]-X1[1])+X1[1]-X3[1]/2.+ 20
    
	return X

def grab(X):

	deltarobot.GripperOpen()	
	deltarobot.DeltaGoPoint(X[0],-X[1],-500)
	deltarobot.DeltaGoPoint(X[0],-X[1],-700)
	deltarobot.GripperClose()
	time.sleep(0.5)
	deltarobot.DeltaGoPoint(X[0],-X[1],-450)
	deltarobot.DeltaGoPoint(0.,250,-450)
	time.sleep(0.5)
	deltarobot.GripperOpen()		
	time.sleep(0.1)
	deltarobot.DeltaGoHome()

	
if __name__ == "__main__":
#    net = load_net("../cfg/yolov3-tiny-obj_big_test.cfg", "../backup-last/yolov3-tiny-obj_big_80000.weights", 0)
	net = load_net(b"../cfg/yolov3-tiny-obj_small_test.cfg", b"../backup/yolov3-tiny-obj_small_50000.weights", 0)
	meta = load_meta(b"../data/obj.data")
#    r = detect(net, meta, "../chestnut.png")

	X1=[0.   ,0.   ,0.   ]
	X2=[420. ,0    ,0.   ]
	X3=[0. ,230.   ,0.   ]
	X =[0.   ,0.   ,0.   ]
	P1=[1,1,0]
	P2=[640,0,0]
	P3=[0,480,0]
    # 640
    # 480
    

#    print r[0],r[0][2][0]
  #  w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	#deltarobot = DeltaRobot()
	#deltarobot.TorqueOn()
	#deltarobot.DeltaGoHome()
	#deltarobot.GripperCheck()

	while(True):
		#deltarobot.DeltaGoHome()
		cap = cv2.VideoCapture(0)
		time.sleep(0.2)
		while(cap.isOpened()==False):	
			cap = cv2.VideoCapture(0)
			#cap.set(CV_CAP_PROP_BUFFERSIZE, 1)
		ret, frame = cap.read()
		cap.release()
		cv2.imwrite('/home/nvidia/Chestnut/check.jpg',frame)
		time.sleep(0.5)		
		cv2.imshow("test",frame)
		while not os.path.exists('/home/nvidia/Chestnut/check.jpg'):
    			time.sleep(0.1)
		if os.path.isfile('/home/nvidia/Chestnut/check.jpg'):
			r = detect(net, meta,  b'/home/nvidia/Chestnut/check.jpg')
			#print ("Test im")
			if len(r)>0:
				print( r)
				time.sleep(0.1)
				#cv2.rectangle(frame, (int(r[0][2][0]-r[0][2][2]/2), int(r[0][2][1]-r[0][2][3]/2)), (int(r[0][2][0]+r[0][2][2]/2), int(r[0][2][1]+r[0][2][3]/2)), (255,0,0), 2)
				#time.sleep(0.1)
				#cv2.imshow("chest nut detection",frame)
				time.sleep(0.5)
				P=[r[0][2][0],r[0][2][1]]
				print("P",P)
				X=coor(X1,X2,X3,P1,P2,P3,P)
				print( "X",X)
				#grab(X)	   
				#os.remove('/home/nvidia/Chestnut/check.jpg')
				#time.sleep(1)
		if cv2.waitKey(1) & 0xFF == ord('q'):
           		break

    

cap.release()
cv2.destroyAllWindows()   
    

