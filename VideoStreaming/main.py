# main.py
# import the necessary packages
from flask import Flask, render_template, Response
from camera import VideoCamera
import tensorflow as tf
from keras import models
import os
import numpy as np
import cv2
from keras.preprocessing import image
import PIL as Image
import matplotlib.pyplot as plt


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Descoped item for eye detection
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


app = Flask(__name__)

@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')


    def __init__(self):
          #capturing video
        self.video = cv2.VideoCapture(0)
    def __del__(self):
        #releasing camera
        self.video.release()


def run_prediction(frame, model):
    img = tf.image.resize(frame, size = [224, 224])/255
    imgtest = image.img_to_array(img)
    imgtest = np.expand_dims(imgtest, axis=0)
    return model.predict(imgtest)

def largest_face(faces):
    x, y, w, h = faces[0]
    facesize = (h -y ) + (w - x)
#    for (x, y, w, h) in faces:
    for xl,yl, wl, hl in faces:
        if ((hl - yl) * (wl - xl)) > facesize:
            x, y, w, h = xl, yl, wl, hl
            facesize =  (hl - yl) + (wl - xl)
    return x, y, w, h


class LoadVideo(object):

    def __init__(self):
          #capturing video
         self.video = cv2.VideoCapture(0)
    def __del__(self):
        #releasing camera
        self.video.release()

def gen(video):

    #os.chdir(r'~\VideoStreaming\model')
    new_model = tf.keras.models.load_model('model/model_save_2/')

    font = cv2.FONT_HERSHEY_SIMPLEX

        # org
    org = (200, 200)

        # fontScale
    fontScale = .7

        # Dark Green color in BGR
    color = (255,255,255)

        # Line thickness of 2 px
    thickness = 2

    # Use urllib to get the image and convert into a cv2 usable format

    while True:

        ret, frame = video.video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        #eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw the rectangle around each face

        if len(faces) == 0:
            mask = 'No Face Detected'
            cv2.putText(frame, mask, (100,400), font, fontScale, color, thickness, cv2.LINE_AA)
        elif len(faces) >= 1:

            x, y, w, h = largest_face(faces)

            im = frame[y:(y+h), x:(x+w)]
            prediction = run_prediction(im, new_model)
            if prediction[0][0] > prediction[0][1]:
                prob = round(float(prediction[0][0]),3)
                mask = 'Mask Detected with Probability: {}%'.format(prob)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, mask, (50,400), font, fontScale, color, thickness, cv2.LINE_AA)
            else:
                prob = round(float(prediction[0][1]),3)
                mask = 'No Mask Detected With Probability: {}%'.format(prob)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, mask, (50,400), font, fontScale, color, thickness, cv2.LINE_AA)
        ret, jpeg = cv2.imencode('.jpg', frame)

        ab = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + ab + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(LoadVideo()),
                   mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
    # defining server ip address and port
    app.run(host='0.0.0.0',port='5000', debug=True)
