from django.shortcuts import render,redirect
from django.http import HttpResponse, HttpResponseRedirect
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
from keras.models import load_model
import pickle
import numpy as np
import shutil
import cv2
from keras.preprocessing import image                  
from PIL import ImageFile     
import time      
from django.conf import settings        
import tensorflow as tf      
from .models import Activity
from .forms import VideoForm
from django.shortcuts import render, redirect
from .forms import ActivityForm
from .models import Activity,Video

from .model_constants import BEST_MODEL_1,BEST_MODEL_2,BEST_MODEL_3,BEST_MODEL_4,BEST_MODEL_5
model_1 = load_model(BEST_MODEL_1)
model_2 = load_model(BEST_MODEL_2)
model_3 = load_model(BEST_MODEL_3)
model_4 = load_model(BEST_MODEL_4)
model_5 = load_model(BEST_MODEL_5)

with open(r"D:\MAJOR_FINAL\major_final\myapp\model\labels_list.pkl", "rb") as handle:
    labels_id = pickle.load(handle)




def index(request):
    return render(request, 'base.html')


def update_activity(request):
    activity = Activity.objects.first()  
    form = ActivityForm(instance=activity)
    if request.method == 'POST':
        form = ActivityForm(request.POST, request.FILES, instance=activity)
        if form.is_valid():
            activity = form.save()
            image_data = activity.file.path  
            
            prediction = return_prediction(image_data)
            return render(request, 'index.html', {'activity': activity, 'prediction': prediction})  
        form = ActivityForm(instance=activity)
        
    return render(request, 'index.html', {'form': form})



def return_prediction(filename):
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True  
    print(type(filename))
    test_tensors = path_to_tensor(filename)
    ypred_test_1 = model_1.predict(test_tensors,verbose=1)
    # ypred_test_2 = model_2.predict(image_tensor,verbose=1)
    # ypred_test_3 = model_3.predict(image_tensor,verbose=1)
    # ypred_test_4 = model_4.predict(image_tensor,verbose=1)
    ypred_test_5 = model_5.predict(test_tensors,verbose=1)


    ypred_test = np.mean([ypred_test_1,ypred_test_5], axis=0) 

    ypred_class = np.argmax(ypred_test)

    print(ypred_class)
    id_labels = dict()
    for class_name,idx in labels_id.items():
        id_labels[idx] = class_name
    print(id_labels)
    ypred_class = int(ypred_class)
    res = id_labels[ypred_class]
    print(res)
  
   
    return (tags.get(ypred_class, "Unknown"))

tags = {
    0: "safe driving",
    1: "texting - right",
    2: "talking on the phone - right",
    3: "texting - left",
    4: "talking on the phone - left",
    5: "operating the radio",
    6: "drinking",
    7: "reaching behind",
    8: "Hair and Makeup",
    9: "talking to passenger"
}                       

def path_to_tensor(img_path):
    img = cv2.imread(img_path)  
    img = img[50:, 120:-50]  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, (224, 224))
    return np.expand_dims(img, axis=0)

    
# FOR VIDEO
def update_video(request):
    video = Video.objects.first()  
    form = VideoForm(instance=video)
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES, instance=video)
        if form.is_valid():
            video = form.save()
            video_path = video.video_file.path  
            OUTPUT_VIDEO_FILE = "media\media\output_video.mp4"
            vs = cv2.VideoCapture(video_path)
            writer = None
            (W, H) = (None, None)

            while True:
                (grabbed, frame) = vs.read()
                if not grabbed:
                    break

                if W is None or H is None:
                    (H, W) = frame.shape[:2]

                output = frame.copy()
                frame = frame[50:, 120:-50]  
                frame = cv2.resize(frame, (224, 224))  
                frame = np.expand_dims(frame, axis=0) 

                label = predict_result(frame)

                text = "activity: {}".format(label)
                cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.25, (0, 255, 0), 5)

                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, 30,
                                              (W, H), True)

                writer.write(output)

                # Display the frame
                cv2.imshow("Output", output)
                key = cv2.waitKey(1) & 0xFF

                # If 'q' is pressed, break from the loop
                if key == ord("q"):
                    break

            print("[INFO] cleaning up...")
            if writer is not None:
                writer.release()
            vs.release()
            cv2.destroyAllWindows()
             
    else:
        form = VideoForm(instance=video)
        
    return render(request, 'index1.html', {'form': form})


def predict_result(image_tensor):
    ypred_test_1 = model_1.predict(image_tensor,verbose=1)
    ypred_test_2 = model_2.predict(image_tensor,verbose=1)
    # ypred_test_3 = model_3.predict(image_tensor,verbose=1)
    # ypred_test_4 = model_4.predict(image_tensor,verbose=1)
    ypred_test_5 = model_5.predict(image_tensor,verbose=1)


    ypred_test = np.mean([ypred_test_1,ypred_test_2,ypred_test_5], axis=0) 

    ypred_class = np.argmax(ypred_test)

    ypred_class = np.argmax(ypred_test,axis=1)
    print(ypred_class)

    id_labels = dict()
    for class_name,idx in labels_id.items():
        id_labels[idx] = class_name
    ypred_class = int(ypred_class)
    print(id_labels[ypred_class])


     
    class_name = dict()
    class_name["c0"] = "SAFE_DRIVING"
    class_name["c1"] = "TEXTING_RIGHT"
    class_name["c2"] = "TALKING_PHONE_RIGHT"
    class_name["c3"] = "TEXTING_LEFT"
    class_name["c4"] = "TALKING_PHONE_LEFT"
    class_name["c5"] = "OPERATING_RADIO"
    class_name["c6"] = "DRINKING"
    class_name["c7"] = "REACHING_BEHIND"
    class_name["c8"] = "HAIR_AND_MAKEUP"
    class_name["c9"] = "TALKING_TO_PASSENGER"


    
    print(ypred_class)
    return (tags.get(ypred_class, "Unknown"))