import time
import datetime
import cv2
from matplotlib import collections
import numpy as np
import csv

class ObjectDetection:
    counter = 0
    value_time = []
    def __init__(self):
        ObjectDetection.counter 
        self.flag = 0
        self.judul = ['Time', 'Counter']
        ObjectDetection.value_time 
        self.MODEL = cv2.dnn.readNet(
            'models/yolov4-custom_best.weights',
            'models/yolov4-custom.cfg'
        )
        self.CLASSES = []
        with open("models/obj.names", "r") as f:
            self.CLASSES = [line.strip() for line in f.readlines()]

        self.OUTPUT_LAYERS = [self.MODEL.getLayerNames()[i - 1] for i in self.MODEL.getUnconnectedOutLayers()]

    def detectObj(self, snap):
        height, width, channels = snap.shape
        blob = cv2.dnn.blobFromImage(snap, 1/255, (416, 416), swapRB=True, crop=False)

        self.MODEL.setInput(blob)
        outs = self.MODEL.forward(self.OUTPUT_LAYERS)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    # Rectangle coordinates
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)                  

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.CLASSES[class_ids[i]])
                color_dict = {"Fokus": (0, 255, 0), "TidakFokus": (0, 0, 255)}
                cv2.rectangle(snap, (x, y), (x + w, y + h), color_dict[label] ,2)
                cv2.putText(snap, label, (x, y - 5), font, 2, color_dict[label], 2)

                # Counting Label TidakFokus
                
                # baca waktu di append dlm satu kolom
                if (label == "TidakFokus"):    
                    time = datetime.datetime.now()
                    if (self.flag == 0):    
                        self.flag = 1
                        ObjectDetection.counter += 1   
                        ObjectDetection.value_time.append([time.strftime('%H:%M'), ObjectDetection.counter])
                        # write to csv
                        f = open('value_time.csv', 'w')
                        writer = csv.writer(f)
                        writer.writerow(self.judul)
                        writer.writerows(ObjectDetection.value_time) 
                        print("ini self.counter objdetect: " + str(ObjectDetection.counter))
                        print(ObjectDetection.value_time)   
                    else:
                        ObjectDetection.counter = ObjectDetection.counter    
                else:
                    if (self.flag == 1):
                        self.flag = 0
                    else:
                        ObjectDetection.counter = ObjectDetection.counter
                cv2.putText(snap, "Tidak Fokus: " + str(ObjectDetection.counter), (500, 50), font, 2, (0,0,0), 2)
                       
        return snap

    def show_counter(self):
        counter = ObjectDetection.counter
        print("ini counter = global counter: " + str(counter))
        return counter    

    def show_chart(self):
        waktu = ObjectDetection.value_time
        print("ini array dari value time: " + str(waktu))
        return waktu

class VideoStreaming(object):
    def __init__(self):
        super(VideoStreaming, self).__init__()
        self.VIDEO = cv2.VideoCapture(0)

        self.MODEL = ObjectDetection()

        self._preview = False
        self._flipH = True
        self._detect = True

    @property
    def preview(self):
        return self._preview

    @preview.setter
    def preview(self, value):
        self._preview = bool(value)

    @property
    def flipH(self):
        return self._flipH

    @flipH.setter
    def flipH(self, value):
        self._flipH = bool(value)

    @property
    def detect(self):
        return self._detect

    @detect.setter
    def detect(self, value):
        self._detect = bool(value)
    
    def show(self):
        while(self.VIDEO.isOpened()):
            ret, snap = self.VIDEO.read()
            if self.flipH:
                snap = cv2.flip(snap, 1)
            
            if ret == True:
                if self._preview:
                    # snap = cv2.resize(snap, (0, 0), fx=0.5, fy=0.5)
                
                    if self.detect:
                        snap = self.MODEL.detectObj(cv2.resize(snap, (800, 450)))

                else:                 
                    snap = np.zeros((
                        int(self.VIDEO.get(cv2.CAP_PROP_FRAME_HEIGHT)-50),
                        int(self.VIDEO.get(cv2.CAP_PROP_FRAME_WIDTH)+120)
                    ), np.uint8)
                    label = 'Camera Disabled'
                    H, W = snap.shape
                    font = cv2.FONT_HERSHEY_PLAIN
                    color = (255, 255, 255)
                    cv2.putText( snap, label, ((W//2 - 130, H//2)), font, 2, color, 2)

                frame = cv2.imencode('.jpg', snap)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.01)
    
            else:
                break
        print('off')
