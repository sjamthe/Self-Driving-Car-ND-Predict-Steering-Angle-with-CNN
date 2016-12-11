"""
Self driving using CNN steering angle prediction
"""
import sys
from scipy.ndimage import imread
from scipy.misc import imresize
from keras import models
import numpy as np

lmodel = None
rmodel = None

def prepare_image(image):
  left = image[64:128,30:94,:]
  left = imresize(left,(32,32),interp='nearest')
  right = image[64:128,226:290,:]
  right = imresize(right,(32,32),interp='nearest')

  return left, right

def predict(model, image):
  X_test = np.array([image])
  results = model.predict(X_test, batch_size=64, verbose=0)
  proba = model.predict_proba(X_test, batch_size=64, verbose=0)
  y_out = np.array(results).argmax(1)

  return y_out[0], proba[0]

def angle(val):
  return (val-8.6)/20

def steering_angle(left_pred, left_prob, right_pred, right_prob):
  if(left_prob[left_pred] > right_prob[right_pred]):
    pred = left_pred
  else:
    pred = right_pred

  return angle(pred)

def get_throttle(speed, angle):
  max_speed = 40
  max_turn = 0.43
  sharp_turn = 0.1
  turn_speed = 12
  max_throttle = 0.2
  min_throttle = 0.2

  if(abs(angle) > sharp_turn):
    if(speed > turn_speed):
      #slow down
      throttle = -1*max_throttle *(abs(angle) - sharp_turn)/(max_turn - sharp_turn)
    else:
      throttle = min_throttle
  else:
    if(speed < max_speed):
      # accelerate
      throttle = max_throttle *(max_speed - speed)/max_speed
    else:
      throttle = -1*max_throttle

  return throttle

def cnn_angle(cnt, image, psteering_angle=0, pthrottle=0, pspeed=0):
  global lmodel, rmodel

  #print("In angle",cnt,pthrottle,psteering_angle)
  left, right = prepare_image(image)
  left_pred, left_prob = predict(lmodel, left)
  right_pred, right_prob = predict(rmodel, right)
  angle = steering_angle(left_pred, left_prob,right_pred, right_prob)
  angle = np.round(angle,2)
  throttle = get_throttle(pspeed, angle)
  #print('returning ', angle, throttle)
  return angle, throttle

def load_models():
  global lmodel, rmodel

  lmodel = models.load_model('left_steering_model.h5')
  rmodel = models.load_model('right_steering_model.h5')
  print("Models loaded.")

def load_image(file):
  image = imread(file)

  return image

if __name__ == '__main__':

  image = load_image(sys.argv[1])
  load_models()
  angle, throttle, out_image = cnn_angle(0, image)
  print(angle, throttle)



