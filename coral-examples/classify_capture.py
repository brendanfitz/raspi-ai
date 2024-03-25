# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo to classify Raspberry Pi camera stream."""
import argparse
import collections
from collections import deque
import common
import io
import numpy as np
import operator
import os
import picamera2
import tflite_runtime.interpreter as tflite
import time
import PIL
import cv2
import pygame

pygame.init()
fart = pygame.mixer.Sound('./building_blocks/fart.mp3')

colour = (255,105,180)
org_x, org_y, org_dy = 0, 70, 70
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 2.5
thickness = 4

Category = collections.namedtuple('Category', ['id', 'score'])

def get_output(interpreter, top_k, score_threshold):
    """Returns no more than top_k categories with score >= score_threshold."""
    scores = common.output_tensor(interpreter, 0)
    categories = [
        Category(i, scores[i])
        for i in np.argpartition(scores, -top_k)[-top_k:]
        if scores[i] >= score_threshold
    ]
    return sorted(categories, key=operator.itemgetter(1), reverse=True)

def class_to_img(request):
    with picamera2.MappedArray(request, 'main') as m:
        for i, result in enumerate(results):
            annotate_text = '{:.0f}% {}'.format(100*result[1], labels[result[0]])
            cv2.putText(m.array, annotate_text, (org_x, org_y+i*org_dy), font, scale, colour, thickness)


def main():
    default_model_dir = './all_models'
    default_model = 'mobilenet_v2_1.0_224_quant_edgetpu.tflite'
    default_labels = 'imagenet_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    args = parser.parse_args()

    global labels
    with open(args.labels, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)

    interpreter = common.make_interpreter(args.model)
    interpreter.allocate_tensors()

    with picamera2.Picamera2() as camera:
        width, height, channels = common.input_image_size(interpreter)
        config = camera.create_video_configuration(
            main=dict(size=(1080, 1080)),
            lores=dict(size=(width, height))
        )
        camera.video_configuration.controls.FrameRate = 30.0
        camera.configure(config)

        camera.post_callback = class_to_img

        camera.start(show_preview=True)
        
        (w1, h1) = camera.stream_configuration("lores")["size"]
        s1 = camera.stream_configuration("lores")["stride"]
        global results
        results = []
        try:
            while True:
                time.sleep(1)
                mb = camera.capture_array("main")
                mb = PIL.Image.fromarray(mb)
                mb = mb.convert('RGB')
                mb = mb.resize((width, height))
                mb.save('main_input_image.jpg')
                mb = np.array(mb)

                # buffer = camera.capture_buffer("lores")
                # grey = buffer[:s1 * h1].reshape((h1, s1))
                # rgb = cv2.cvtColor(grey,cv2.COLOR_GRAY2RGB)
                # rgb = cv2.resize(rgb, (width, height))
                # PIL.Image.fromarray(rgb).save('input_image.jpg')
                input_data = np.expand_dims(mb, axis=0)
                common.input_tensor(interpreter)[:,:] = np.reshape(input_data, common.input_image_size(interpreter))
                interpreter.invoke()
                results = get_output(interpreter, top_k=3, score_threshold=0)
                result_labels = [labels[result[0]] for result in results]

                if 'banana' in result_labels:
                    playing = fart.play()
                    while playing.get_busy():
                        pygame.time.delay(100)
        finally:
            camera.stop()


if __name__ == '__main__':
    main()
