from cProfile import label
from tkinter.tix import Tree
from django.http import HttpResponse
from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2, time, json
import torch
import numpy as np
import os
from django.conf import settings


model = torch.hub.load('ultralytics/yolov5', 'custom', os.path.join(settings.BASE_DIR,'models\\best.pt'))  # custom trained model

classes = ('Mu', 'Quan_Ao', 'Gang_Tay') # Thay thế bằng classes của mô hình train

detected = False

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.colors = np.random.uniform(0, 255, size=(len(classes), 3))

    def __del__(self):
        self.video.release()

    def get_frame(self):
        
        success, frame = self.video.read()
        if not success:
            return None

        results = model(frame)
        
        for box in results.xyxy[0]: 
            
            if box[5] is not None and (float(box[4]) * 100) >= 50:
                
                global xA, yA, xB, yB, label, accuracy, detected, curr_time, full_info
                xB = int(box[2])
                xA = int(box[0])
                yB = int(box[3])
                yA = int(box[1])
                accuracy = round(float(box[4] * 100),2)
                label = classes[int(box[5])]
                curr_time = time.strftime("%H:%M:%S", time.localtime())
                full_info = results.pandas().xyxy[0].values.tolist()
            
                frame = cv2.rectangle(frame, (xA, yA), (xB, yB), self.colors[int(box[5])], 1)
                frame = cv2.putText(frame, classes[int(box[5])], (xA, yA - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                detected = True
                
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

def info_detect():
    
    global detected
    
    if detected == True:
        detected = False
        yield 'data: {0}/{1}/{2}/{3}/{4}/{5}/{6}/{7}\n\n'.format(xA,yB,xB,yB,accuracy,label,curr_time,full_info)
        

def send_info(request):
    return  StreamingHttpResponse(info_detect(), content_type='text/event-stream')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()), content_type='multipart/x-mixed-replace; boundary=frame')

    
def index(request):
    return render(request, 'yolo_app/index.html')

