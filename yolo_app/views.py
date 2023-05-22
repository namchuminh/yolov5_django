from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
import torch
import numpy as np
import os
from django.conf import settings


model = torch.hub.load('ultralytics/yolov5', 'custom', os.path.join(settings.BASE_DIR,'models\\best.pt'))  # custom trained model

classes = ('Mu', 'Quan_Ao', 'Gang_Tay') # Thay thế bằng classes của mô hình train

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
            if box[5] is not None:
                xB = int(box[2])
                xA = int(box[0])
                yB = int(box[3])
                yA = int(box[1])
                
                frame = cv2.rectangle(frame, (xA, yA), (xB, yB), self.colors[int(box[5])], 1)
                frame = cv2.putText(frame, classes[int(box[5])], (xA, yA - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

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
