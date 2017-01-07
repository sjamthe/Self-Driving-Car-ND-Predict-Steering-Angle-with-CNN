"""
I am using the same CNN model I built for traffic sign prediction. I have only
changed the last layer. Instead of predicting 47 classes of traffic signs it
predicts 18 classes of steering angles. A valid steering angle turn is between
0.43 radians in either direction (which is 25 degrees).
-0.43 becomes class 0 and .43 becomes class 17

The benefit of using the same CNN function is that we can load the weights
from traffic model and reduce training time.

The same network is used for left and right images but is trained separately.
both networks predict steering angle. In training
"""
import sys
import numpy as np
import os.path
import pickle
from scipy.ndimage import imread
from scipy.misc import imresize
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten
from keras.layers.pooling import MaxPooling2D,AveragePooling2D
from keras.layers.core import Dropout
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator

"""
Load the data prepared by prep-data.py
each image is 32x32 and label is 0-17
"""
def load_data(picklefile, in_data):
  with open(picklefile, mode='rb') as f:
    data = pickle.load(f)
    print("Read pickle data left_images", data['left_images'].shape,
    'right_images',data['right_images'].shape,
    'labels',data['labels'].shape)

    if(in_data is not None):
      data['left_images'] = np.append(in_data['left_images'],data['left_images'], axis=0)
      data['right_images'] = np.append(in_data['right_images'],data['right_images'], axis=0)
      data['labels'] = np.append(in_data['labels'],data['labels'], axis=0)

  return data

def build_model():
  input_shape = (32,32,3)
  model = Sequential()
  model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape,
                          activation='relu',W_regularizer=l2(0.001)))
  model.add(MaxPooling2D())
  model.add(Convolution2D(64, 3, 3, border_mode='same',activation='relu',W_regularizer=l2(0.001)))
  model.add(Convolution2D(64, 3, 3, border_mode='same',activation='relu',W_regularizer=l2(0.001)))
  model.add(MaxPooling2D())
  model.add(Flatten())
  model.add(Dense(512, name="hidden1",activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(18,name="new-output",activation='softmax'))

  #old weights from traffic model
  model.load_weights('my_keras_traffic_model.h5', by_name=True)
  return model

"""
Function to shuffle the input images and output data randomly
"""
def unison_shuffled_copies(a, b, c):
  assert len(a) == len(b)
  assert len(a) == len(c)
  p = np.random.permutation(len(a))
  return a[p], b[p], c[p]

"""
Train and validate the model.
we reserve 10% for validation
"""
def train_model(model, X_input, Y_output):
  learning_rate = .0001

  opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

  #[X_input, Y_output] = unison_shuffled_copies(X_input, Y_output)
  Y_output = np.uint8(np.round((Y_output*20+8.6),0))
  print(np.min(Y_output),np.max(Y_output))

  X_train, X_test, y_train, y_test = train_test_split(X_input, Y_output,
                                         test_size=0.1, random_state=11)

  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                         test_size=0.1, random_state=42)

  y_train = np_utils.to_categorical(y_train, 18)
  y_val = np_utils.to_categorical(y_val, 18)

  batch_size = 64
  nb_epoch = 20

  hist = model.fit(X_train, y_train,
                      batch_size=batch_size, nb_epoch=nb_epoch,
                      validation_data=(X_val,y_val), verbose=1)

  #print(hist.history)

  return X_test, y_test

"""
Convert the output label back to steering angle in radians
-0.43 is class 0 and .43 is 17
"""
def angle(val):
  return (val-8.6)/20

def predict(model, X_test, y_test):
  y1_test = np_utils.to_categorical(y_test, 18)
  ret = model.evaluate(X_test, y1_test)

  for num in range(len(ret)):
    print(model.metrics_names[num],'=',ret[num])

  results = model.predict(X_test, batch_size=64, verbose=1)
  proba = model.predict_proba(X_test, batch_size=64, verbose=1)
  y_out = np.array(results).argmax(1) #argmax converts to_categorical
  #print (y_out[0:10])
  #print (y_test[0:10])

  return y_out, proba

"""
Compare steering_angle predictions from left and right models
"""
def compare(y_test,left_pred,left_prob,y_right,right_pred,right_prob):
  incorrects = []
  y_errors = []
  for i,test in enumerate(y_test):
    assert y_test[i] == y_right[i]
    if(left_pred[i] != test or test != right_pred[i]):
      #we decide to go with the answer with highest probability
      # the see if we get it right
      if(left_prob[i][left_pred[i]] > right_prob[i][right_pred[i]]):
        pred = left_pred[i]
      else:
        pred = right_pred[i]

      if(pred != test):
        print ("image = ",i, "correct angle = ",angle(test),
             "left pred = ",angle(left_pred[i]),
             "right pred = ",angle(right_pred[i]))
        #print ("probability of correct class = ", proba[i][test], " incorrect = ", proba[i][left_pred[i]])
        incorrects.append([i, test, left_pred[i]])
        y_errors.append(test)

  print("mismatches = ", len(incorrects), "accuracy = ",1-len(incorrects)/len(y_test))

  # Display how many errors for each class of image
  errors = np.bincount(np.array(y_errors))
  ys = np.bincount(np.array(y_test))
  print ("Error distribution:", errors)
  print ("Data distribution", ys)
  #for i,test in enumerate(ys):
    #print(i,test,errors[i],int(100*errors[i]/test))

"""
argv[1] = <dirname> where center camera images are stored. Images filenames are <image_cnt>.jpg
argv[2] = <inputfile> The file is CSV created from mydrive.py
The columns are 'Steering, <image_cnt>,<input steering_angle<,<input throttle>,<input speed>,<output steering_angle>,<output throttle>'
This model uses only image as input and predicts output steering angle
"""
if __name__ == '__main__':

    data = None
    for args in sys.argv[1:]:
      picklefile = args
      data = load_data(picklefile, data)

    lmodel = build_model()
    rmodel = build_model()

    data['labels'], data['left_images'], data['right_images'] = \
      unison_shuffled_copies(data['labels'], data['left_images'], data['right_images'])
    X_left, y_left = train_model(lmodel, data['left_images'], data['labels'])
    lmodel.save('left_steering_model.h5')
    X_right, y_right = train_model(rmodel, data['right_images'], data['labels'])
    rmodel.save('right_steering_model.h5')
    left_pred, left_prob = predict(lmodel, X_left, y_left)
    right_pred, right_prob = predict(rmodel, X_right, y_right)
    compare(y_left,left_pred,left_prob,y_right,right_pred,right_prob)
