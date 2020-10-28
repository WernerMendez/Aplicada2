import csv
import board
import busio as io
import adafruit_mlx90614
import cv2
import RPi.GPIO as GPIO
import datetime
from time import *
import numpy as np
from PIL import Image
from imutils.video import WebcamVideoStream
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.tensorflow_loader import load_tf_model, tf_inference

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(24, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(16, GPIO.OUT)
GPIO.setup(20, GPIO.OUT)
GPIO.setup(21, GPIO.OUT)
i2c = io.I2C(board.SCL, board.SDA, frequency=100000)
mlx = adafruit_mlx90614.MLX90614(i2c)
MASK = 0
check1 = 0
check2 = 0
Access = False
Acceso = str


sess, graph = load_tf_model('models/face_mask_detection.pb')
# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}

def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(260, 260),
              draw_result=True,
              show_result=True
              ):
    global MASK
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  
    image_exp = np.expand_dims(image_np, axis=0)
    y_bboxes_output, y_cls_output = tf_inference(sess, graph, image_exp)

    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
                MASK = 1
            else:
                color = (0, 0, 255)
                MASK = 0
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

    if show_result:
        Image.fromarray(image).show()
    return output_info

def getTemp():
    OT=0
    for i in range (0,20):
        OT = mlx.object_temperature + OT
        sleep(0.001)
    OT = (OT/20)+3.5
    return OT

def Fecha():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    today = str(today)
    return today

def Hora():
    now = datetime.datetime.now().strftime("%H:%M:%S")
    now = str(now)
    return now

def Mascarilla():
    global MASK
    if MASK ==1:
        mask = 'Yes'
    else:
        mask = 'No'
    return mask

def CSV():
    with open('/home/pi/Desktop/Apli2/Results.csv',mode='a')as results_read:
        results_write=csv.writer(results_read,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        write_log=results_write.writerow([Fecha(),Hora(),getTemp(),Mascarilla(),Acceso])
        return write_log

cap = WebcamVideoStream(src=0).start()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
status = True
while status:
    img_raw = cap.read()
    img_raw = img_raw[0:640,190:460]
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    if (status):
        inference(img_raw,conf_thresh=0.5,iou_thresh=0.5,target_shape=(260, 260),draw_result=True,show_result=False)
    cv2.putText(img_raw,"%.2f C" % getTemp(),(60,450),cv2.FONT_HERSHEY_SIMPLEX, 1.3,(255,255,255))
    if (getTemp()>35 and getTemp()<37 and MASK ==1):
        GPIO.output(20,True)
        GPIO.output(21,False)
        check1 = check1 + 1
        check2 = 0
        if check1 == 3:
            Acceso = 'Yes'
            CSV()
            Access = True
            check1 = 0
    elif(getTemp()< 35):
        GPIO.output(20,False)
        GPIO.output(21,False)
        check1 = 0
        check2 = 0
        Access = False
    elif(getTemp()> 42):
        GPIO.output(20,False)
        GPIO.output(21,False)
        check1 = 0
        check2 = 0
        Access = False
    else:
        GPIO.output(21,True)
        GPIO.output(20,False)
        check1 = 0
        check2 = check2 + 1
        if check2 == 3:
            Acceso = 'No'
            CSV()
            check2 = 0
    # chequeo de datos y activa dispensador
    while Access :
        Dispensador = GPIO.input(24)
        if Dispensador == 0:
            GPIO.output(16,True)
            sleep(0.4)
            GPIO.output(16,False)
            sleep(3)
            Access = False
        
    cv2.imshow('Mask Detector',img_raw)
    cv2.waitKey(1)

cv2.destroyAllWindows()
cap.stop()
GPIO.cleanup()