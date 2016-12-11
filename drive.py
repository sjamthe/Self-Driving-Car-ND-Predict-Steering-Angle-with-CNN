import argparse
import base64
import json
import os

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
import cnnselfdrive as cnn

import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

def array2PIL(data):
  mode = 'RGB'
  size = data.shape[1],data.shape[0]
  return Image.frombuffer(mode, size, data.tostring(), 'raw', mode, 0, 1)

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
angles = None
prev_image = 0
cnt=0
debug=1
fo = None

@sio.on('telemetry')
def telemetry(sid, data):
    global cnt
    global angles
    global fo

    # The current steering angle of the car
    steering_angle = np.round(float(data["steering_angle"]),2)
    # The current throttle of the car
    throttle = np.round(float(data["throttle"]),2)
    # The current speed of the car
    speed = np.round(float(data["speed"]),0)
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    #print ("shape = ",image_array.shape)
    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    #steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    #steering_angle=angles[cnt]

    cnt+=1
    new_steering_angle, new_throttle = cnn.cnn_angle(cnt, image_array,
            steering_angle, throttle, speed)
    new_steering_angle = np.round(new_steering_angle,2)
    new_throttle = np.round(new_throttle,2)
    #capture result angle
    output = str(cnt) + "," + str(steering_angle)+ "," + \
        str(throttle) + "," + str(speed)+ "," + \
        str(new_steering_angle) +  "," + str(new_throttle)

    #fo.write(output + "\n")

    #print("steering,",output)
    img1 = array2PIL(image_array)
    dir = './input/'
    if(os.path.isdir(dir) is False):
        os.mkdir(dir)
    img1.save(dir + str(cnt) + '.jpg')

    send_control(new_steering_angle, new_throttle)


@sio.on('connect')
def connect(sid, environ):
    global fo

    print("connect ", sid)
    cnt = 0
    #if(fo is not None):
    #    fo.close()
    #fo = open("./cnn-output.log", "w+")
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')

    cnt = 0
    #load models for left image and right image
    cnn.load_models()
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
