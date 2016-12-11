import cv2
import sys
import numpy as np

def draw_square(img, vertices_left,vertices_top,vertices_bottom,vertices_right):
  color=[0, 255, 0]
  thickness=2
  cv2.line(img, (vertices_left,vertices_top),
            (vertices_left,vertices_bottom), color, thickness)
  cv2.line(img, (vertices_left,vertices_top),
            (vertices_right,vertices_top), color, thickness)
  cv2.line(img, (vertices_right,vertices_top),
            (vertices_right,vertices_bottom), color, thickness)
  cv2.line(img, (vertices_right,vertices_bottom),
            (vertices_left,vertices_bottom), color, thickness)
  return img


def draw_left_square(img):
  vertices_left = 30
  vertices_top = 64
  vertices_bottom = 64+vertices_top
  vertices_right = 64+vertices_left

  return draw_square(img, vertices_left,vertices_top,vertices_bottom,vertices_right)

def draw_right_square(img):
  vertices_left = 226
  vertices_top = 64
  vertices_bottom = 64+vertices_top
  vertices_right = 64+vertices_left
  return draw_square(img, vertices_left,vertices_top,vertices_bottom,vertices_right)


if __name__ == '__main__':

  imagefile = sys.argv[1]
  image = cv2.imread(imagefile)
  image = draw_left_square(image)
  image = draw_right_square(image)

  cv2.imwrite('output.jpg',image)
