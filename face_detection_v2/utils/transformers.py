import tensorflow as tf
from utils.extractors import Pathways, load_image, load_labels

DATA_SUBSETS = ('train', 'test', 'val')

def create_image_ds(data_subset):
    if data_subset not in DATA_SUBSETS:
        raise ValueError()
    
    image_ds = tf.data.Dataset.list_files(str(Pathways.aug_sets_path / data_subset / 'images'/ '*.jpg'), shuffle=False)
    image_ds = image_ds.map(load_image)
    image_ds = image_ds.map(lambda x: tf.image.resize(x, (120,120)))
    image_ds = image_ds.map(lambda x: x/255)
    
    return image_ds

def create_label_ds(data_subset):
    if data_subset not in DATA_SUBSETS:
        raise ValueError()

    label_ds = tf.data.Dataset.list_files(str(Pathways.aug_sets_path / data_subset / 'labels'/ '*.json'), shuffle=False)
    label_ds = label_ds.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    return label_ds

def create_fn_ds(data_subset):
    fn_ds = tf.data.Dataset.list_files(str(Pathways.aug_sets_path / data_subset / 'images'/ '*.jpg'), shuffle=False)
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

    if prefetch:
        ds = ds.prefetch(prefetch)

    return ds