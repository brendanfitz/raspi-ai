# source: https://www.raspberrypi.com/news/using-the-picamera2-library-with-tensorflow-lite/

import tflite_runtime.interpreter as tflite

import sys
import os
import argparse

import cv2
import numpy as np
from PIL import Image
from PIL import ImageFont, ImageDraw

from picamera2 import *

normalSize = (640, 480)
lowresSize = (320, 240)

rectangles = []

def ReadLabelFile(file_path):
  with open(file_path, 'r') as f:
    lines = f.readlines()
  ret = {}
  for line in lines:
    pair = line.strip().split(maxsplit=1)
    ret[int(pair[0])] = pair[1].strip()
  return ret

def DrawRectangles(request):
   with MappedArray(request, "main") as m:
       im = np.array(m.array, copy=False, dtype=np.uint8).reshape((normalSize[1],normalSize[0], 4))

       for rect in rectangles:
          print(rect)
          rect_start = (int(rect[0]*2) - 5, int(rect[1]*2) - 5)
          rect_end = (int(rect[2]*2) + 5, int(rect[3]*2) + 5)
          cv2.rectangle(im, rect_start, rect_end, (0,255,0,0))
          if len(rect) == 5:
            text = rect[4]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im, text, (int(rect[0]*2) + 10, int(rect[1]*2) + 10), font, 1, (255,255,255),2,cv2.LINE_AA)
       del im

def InferenceTensorFlow( image, model, output, label=None):
   global rectangles

   if label:
       labels = ReadLabelFile(label)
   else:
       labels = None

   interpreter = tflite.Interpreter(model_path=model, num_threads=4)
   interpreter.allocate_tensors()

   input_details = interpreter.get_input_details()
   output_details = interpreter.get_output_details()
   height = input_details[0]['shape'][1]
   width = input_details[0]['shape'][2]
   floating_model = False
   if input_details[0]['dtype'] == np.float32:
       floating_model = True

   rgb = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
   initial_h, initial_w, channels = rgb.shape

   picture = cv2.resize(rgb, (width, height))

   input_data = np.expand_dims(picture, axis=0)
   if floating_model:
      input_data = (np.float32(input_data) - 127.5) / 127.5

   interpreter.set_tensor(input_details[0]['index'], input_data)

   interpreter.invoke()

   detected_boxes = interpreter.get_tensor(output_details[0]['index'])
   detected_classes = interpreter.get_tensor(output_details[1]['index'])
   detected_scores = interpreter.get_tensor(output_details[2]['index'])
   num_boxes = interpreter.get_tensor(output_details[3]['index'])

   rectangles = []
   for i in range(int(num_boxes)):
      top, left, bottom, right = detected_boxes[0][i]
      classId = int(detected_classes[0][i])
      score = detected_scores[0][i]
      if score > 0.5:
          xmin = left * initial_w
          ymin = bottom * initial_h
          xmax = right * initial_w
          ymax = top * initial_h
          box = [xmin, ymin, xmax, ymax]
          rectangles.append(box)
          if labels:
              print(labels[classId], 'score = ', score)
              rectangles[-1].append(labels[classId])
          else:
              print ('score = ', score)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path of the detection model.', required=True)
    parser.add_argument('--label', help='Path of the labels file.')
    parser.add_argument('--output', help='File path of the output image.')
    args = parser.parse_args()

    if ( args.output):
      output_file = args.output
    else:
      output_file = 'out.jpg'

    if ( args.label ):
      label_file = args.label
    else:
      label_file = None

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": normalSize},
                                                 lores={"size": lowresSize, "format": "YUV420"})
    picam2.configure(config)

    stride = picam2.stream_configuration("lores")["stride"]
    picam2.request_callback = DrawRectangles

    picam2.start(show_preview=True)

    while True:
        buffer = picam2.capture_buffer("lores")
        grey = buffer[:stride*lowresSize[1]].reshape((lowresSize[1], stride))
        result = InferenceTensorFlow( grey, args.model, output_file, label_file )

if __name__ == '__main__':
  main()