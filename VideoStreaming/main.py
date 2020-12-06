# main.py
# import the necessary packages
from flask import Flask, render_template, Response
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

# Initialize the Flask html homepage and begin VideoCapture.
def index():
    # rendering webpage
    return render_template('index.html')


    def __init__(self):
          #capturing video
        self.video = cv2.VideoCapture(0)
    def __del__(self):
        #releasing camera
        self.video.release()

# Takes an image and a model and runs a prediction. The image is resized and normalized and transformed into a format
# appropriate as an input for the model.predict function. The results of model.predict are returned.
def run_prediction(frame, model):
    img = tf.image.resize(frame, size = [224, 224])/255
    imgtest = image.img_to_array(img)
    imgtest = np.expand_dims(imgtest, axis=0)
    return model.predict(imgtest)

# Function designed to iterate through Faces returned from CV2 face detection. Only the largest face in the
# image based on the returned dimensions is used for model prediction

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
    new_model = tf.keras.models.load_model('model/mask-classifier/')

    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (200, 200)

    # fontScale
    fontScale = .7

    # White color in BGR
    color = (255,255,255)

    # Line thickness of 2 px
    thickness = 2

    # Loops through camera frames to read and process image data for prediction
    while True:

        ret, frame = video.video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect image faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        #eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

        # If no faces are identified overlay a message to the user indicating a prediction cannot be made
        if len(faces) == 0:
            mask = 'No Face Detected'
            cv2.putText(frame, mask, (100,400), font, fontScale, color, thickness, cv2.LINE_AA)
        elif len(faces) >= 1: # If at least one face is returned proceed with making a prediction
            x, y, w, h = largest_face(faces) # Perform a prediction only on the largest face

            im = frame[y:(y+h), x:(x+w)] #Reduce the input for image prediction to just the portion of the image containing a face
            prediction = run_prediction(im, new_model) # Call the prediction fuction

            if prediction[0][0] > prediction[0][1]: # If a mask is predicted overlay a message to the user
                prob = round(float(prediction[0][0]*100),2)
                mask = 'Mask Detected with Probability: {}%'.format(prob)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, mask, (50,400), font, fontScale, color, thickness, cv2.LINE_AA)
            else:
                prob = round(float(prediction[0][1]*100),2) # If no mask is predicted overlay a message to the user
                mask = 'No Mask Detected With Probability: {}%'.format(prob)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, mask, (50,400), font, fontScale, color, thickness, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', frame) # Convert frame to jpg

        ab = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + ab + b'\r\n\r\n') # Return a response to display


@app.route('/video_feed') # Provides continous video feed with prediction details to web application
def video_feed():
    return Response(gen(LoadVideo()),
                   mimetype='multipart/x-mixed-replace; boundary=frame')



# Starts the Flask application accessible from the device IP address in the internal network.
if __name__ == '__main__':
    # defining server ip address and port
    app.run(host='0.0.0.0',port='5000', debug=True)
