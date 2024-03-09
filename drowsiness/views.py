from django.shortcuts import render
from django.http import HttpResponse
import cv2
import dlib
from scipy.spatial import distance as dist
from playsound import playsound
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
import os
import pygame 
model = tf.keras.models.load_model(r'C:\Users\aashi\Desktop\MAJOR_FINAL\major_final\drowsiness\drowsy.h5')

scalar = joblib.load(r'C:\Users\aashi\Desktop\MAJOR_FINAL\major_final\drowsiness\scaler1.pkl')
EAR_THRESHOLD = 0.3
EAR_CONSEC_FRAMES = 30
ALARM_SOUND_PATH = (r"C:\\Users\\aashi\\Desktop\\MAJOR_FINAL\\major_final\\media\\media\\alarm.WAV")
def eye_aspect_ratio(eye):
    x = [point.x for point in eye]
    y = [point.y for point in eye]
    A = np.linalg.norm(np.array([x[1] - x[5], y[1] - y[5]]))
    B = np.linalg.norm(np.array([x[2] - x[4], y[2] - y[4]]))
    C = np.linalg.norm(np.array([x[0] - x[3], y[0] - y[3]]))
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate PUC
def pupil_to_eye_center_distance(eye):
    x = [point.x for point in eye]
    y = [point.y for point in eye]
    d = np.linalg.norm(np.array([x[0] - x[3], y[0] - y[3]]))
    return d

# Function to calculate MAR
def mouth_aspect_ratio(mouth):
    x = [point.x for point in mouth]
    y = [point.y for point in mouth]
    A = np.linalg.norm(np.array([x[13] - x[19], y[13] - y[19]]))
    B = np.linalg.norm(np.array([x[14] - x[18], y[14] - y[18]]))
    C = np.linalg.norm(np.array([x[15] - x[17], y[15] - y[17]]))
    mar = (A + B + C) / (3.0 * np.linalg.norm(np.array([x[12] - x[16], y[12] - y[16]])))
    return mar

# Function to calculate MOE
def mouth_to_eye_ratio(eye, mouth):
    ear = eye_aspect_ratio(eye)
    mar = mouth_aspect_ratio(mouth)
    if ear == 0:  # Avoid division by zero
        return 0
    moe = mar / ear
    return moe

# define constants for drowsiness detection


def drowsy(request):
    def detect_drowsiness():
        drowsy_frames = 0
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("drowsiness\shape_predictor_68_face_landmarks.dat")

        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively
        

        # initialize the video stream and sleep counter
        cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS)

        print("Frame rate of the webcam:", fps)
        pygame.mixer.init()

        # Load the sound file
        alarm_sound = pygame.mixer.Sound(ALARM_SOUND_PATH)
        sleep_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            rects = detector(gray, 0)

            # loop over the face detections
            for rect in rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy array
                shape = predictor(gray, rect)

            # extract the coordinates of the facial landmarks
                points = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
                ear = eye_aspect_ratio(shape.parts()[36:42])
                puc = pupil_to_eye_center_distance(shape.parts()[36:42])
                mar = mouth_aspect_ratio(shape.parts()[48:68])
                moe = mouth_to_eye_ratio(shape.parts()[36:42], shape.parts()[48:68])
                
                X = [ear,mar]
              

                print(X)

                # X = scalar.transform(X)
                
                array_2d = np.array(X).reshape(1, 2)
                array_2d = scalar.transform(array_2d)
                array_2d = array_2d.reshape(1, 2, 1, 1)     
  
                # Y.reshape(-1,1)
                prediction = model.predict(array_2d)
                print(prediction)
                prediction = np.round(prediction)

                print(prediction)
                if prediction == 1:
                    drowsy_frames += 1
                else:
                    drowsy_frames = 0

                if drowsy_frames >= EAR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    alarm_sound.play()
                    print("Drowsiness Detected")

                # draw the computed eye aspect ratio on the frame for visualization
                # cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # cv2.putText(frame, "PUC: {:.2f}".format(puc), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # cv2.putText(frame, "MOE: {:.2f}".format(moe), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)




            # display the frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # do a bit of cleanup
        cap.release()
        cv2.destroyAllWindows()

    # Call the function to detect drowsiness
    detect_drowsiness()

    return render(request, 'based.html')