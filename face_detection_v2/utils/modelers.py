import tensorflow as tf


def localization_loss(y_classes, y_coords, yhat_coords):
    yhat_coords = tf.concat(
        (yhat_coords[:, :4] * tf.cast(y_classes[:, 0:1], tf.float32),
         yhat_coords[:, 4:] * tf.cast(y_classes[:, 1:2], tf.float32)),
        axis=1
    )
    
    # differences between coordinates
    delta_coord = tf.reduce_sum(tf.square(y_coords[:,:2] - yhat_coords[:,:2])) + tf.reduce_sum(tf.square(y_coords[:,4:6] - yhat_coords[:,4:6]))

    # differences between size
    delta_size = 0
    for i in range(2):
        h_true = y_coords[:,3+4*i] - y_coords[:,1+4*i] 
        w_true = y_coords[:,2+4*i] - y_coords[:,0+4*i] 
    
        h_pred = yhat_coords[:,3+4*i] - yhat_coords[:,1+4*i] 
        w_pred = yhat_coords[:,2+4*i] - yhat_coords[:,0+4*i] 
        
        delta_size += tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))
    
    return (delta_coord + delta_size) / tf.cast(tf.shape(y_classes)[0], tf.float32)