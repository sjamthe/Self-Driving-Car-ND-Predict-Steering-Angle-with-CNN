import sys
import numpy as np
import os.path
import pickle
from scipy.ndimage import imread
from scipy.misc import imresize


def load_images(dirname, inputfile, picklefile):
  #read only image_cnt and 'output steering angle' columns
  inputdata = np.loadtxt(inputfile, delimiter=',',usecols=(0,5))
  # We have some angles that are extreme due to bug. remove that data from training set.

  Y_output = []
  left_images = []
  right_images = []
  max_angle = 0.43 #"25 degrees"
  for cnt,angle in inputdata:
    filename = dirname +'/' + str(int(cnt)) + '.jpg'
    if(angle <= max_angle and angle >= -1*max_angle and
      os.path.exists(filename)):
      image = imread(filename)
      #cropping image top & bottom - height is 160 we keep only 64-128
      left = image[64:128,30:94,:]
      left = imresize(left,(32,32),interp='nearest')
      right = image[64:128,226:290,:]
      right = imresize(right,(32,32),interp='nearest')

      left_images.append(left)
      right_images.append(right)
      Y_output.append(angle)
    else:
      print("skipping image",cnt,angle)

  left_images= np.array(left_images)
  right_images= np.array(right_images)
  Y_output = np.array(Y_output)

  data = {'left_images': left_images, 'right_images': right_images, 'labels': Y_output}
  pickle.dump(data,open(picklefile,'wb'))

  print("Created pickle data left_images", data['left_images'].shape,
    'right_images',data['right_images'].shape,
    'labels',data['labels'].shape)

def load_data(picklefile):
  with open(picklefile, mode='rb') as f:
    data = pickle.load(f)
  print("Read pickle data left_images", data['left_images'].shape,
    'right_images',data['right_images'].shape,
    'labels',data['labels'].shape)

  return data

"""
argv[1] = <dirname> where center camera images are stored. Images filenames are <image_cnt>.jpg
argv[2] = <inputfile> The file is CSV created from mydrive.py
The columns are 'Steering, <image_cnt>,<input steering_angle<,<input throttle>,<input speed>,<output steering_angle>,<output throttle>'
This model uses only image as input and predicts output steering angle
"""
if __name__ == '__main__':

    if(len(sys.argv) == 4):
      dirname = sys.argv[1]
      inputfile = sys.argv[2]
      picklefile = sys.argv[3]
      load_images(dirname, inputfile, picklefile)
    else:
      picklefile = sys.argv[1]
      data = load_data(picklefile)
