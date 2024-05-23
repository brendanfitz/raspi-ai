import tensorflow as tf


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