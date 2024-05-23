from pathlib import Path
import tensorflow as tf
import json
import itertools


class Pathways:
    data_wd = Path('/data') / 'raspi_face_detection'
    img_path = data_wd / 'images'
    labels_path = data_wd / 'labels'
    
    model_sets_path = data_wd / 'model_sets'
    aug_sets_path = data_wd / 'aug_data'


def load_image(x): 
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

def load_labels(label_path):
    if isinstance(label_path, tf.Tensor):
        label_path = label_path.numpy()
        
    with open(label_path, 'r', encoding = "utf-8") as f:
        label_data = json.load(f)

    labels = [int(x in label_data['classes']) for x in range(1, 3)]
    labels = [1 if sum(labels) > 0 else 0]

    bboxes = list(itertools.chain(*label_data['bboxes']))

    return labels, bboxes