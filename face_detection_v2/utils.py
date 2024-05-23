from pathlib import Path
import tensorflow as tf
import json
import itertools

data_wd = Path('/data') / 'raspi_face_detection'
img_path = data_wd / 'images'
labels_path = data_wd / 'labels'

model_sets_path = data_wd / 'model_sets'
aug_sets_path = data_wd / 'aug_data'

DATA_SUBSETS = ('train', 'test', 'val')


def load_image(x): 
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)

    return [int(x in label['classes']) for x in range(1, 3)], list(itertools.chain(*label['bboxes']))

def create_image_ds(data_subset):
    if data_subset not in DATA_SUBSETS:
        raise ValueError()
    
    image_ds = tf.data.Dataset.list_files(str(aug_sets_path / data_subset / 'images'/ '*.jpg'), shuffle=False)
    image_ds = image_ds.map(load_image)
    image_ds = image_ds.map(lambda x: tf.image.resize(x, (120,120)))
    image_ds = image_ds.map(lambda x: x/255)
    
    return image_ds

def create_label_ds(data_subset):
    if data_subset not in DATA_SUBSETS:
        raise ValueError()

    label_ds = tf.data.Dataset.list_files(str(aug_sets_path / data_subset / 'labels'/ '*.json'), shuffle=False)
    label_ds = label_ds.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    return label_ds

def create_fn_ds(data_subset):
    fn_ds = tf.data.Dataset.list_files(str(aug_sets_path / data_subset / 'images'/ '*.jpg'), shuffle=False)
    return fn_ds

def create_ds(data_subset, batch_size=128, prefetch=4, shuffle=True, include_fns=False):

    image_ds = create_image_ds(data_subset)
    label_ds = create_label_ds(data_subset)

    if include_fns:
        fn_ds = create_fn_ds(data_subset)
        ds = tf.data.Dataset.zip((image_ds, label_ds, fn_ds))
    else:
        ds = tf.data.Dataset.zip((image_ds, label_ds))
        
    if shuffle:
        ds = ds.shuffle(len(ds))

    ds = ds.batch(batch_size)
    ds = ds.prefetch(prefetch)

    return ds
    


def localization_loss(y_true, yhat, loss_delta_coord_multiplier=1):
    # differences between coordinates
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2])) + tf.reduce_sum(tf.square(y_true[:,4:6] - yhat[:,4:6]))

    # differences between size
    delta_size = 0
    for i in range(2):
        h_true = y_true[:,3+4*i] - y_true[:,1+4*i] 
        w_true = y_true[:,2+4*i] - y_true[:,0+4*i] 
    
        h_pred = yhat[:,3+4*i] - yhat[:,1+4*i] 
        w_pred = yhat[:,2+4*i] - yhat[:,0+4*i] 
        
        delta_size += tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))
    
    return delta_coord * loss_delta_coord_multiplier + delta_size