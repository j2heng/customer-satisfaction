from keras.layers import Lambda   
from keras import backend as K 

# Define Multi-class F1 score

def f1_micro(y_true, y_pred):
    """ Calculate Micro average F1-score for multi-class classification 
    Exemple : 
        y_true = K.constant([[0,0,1], [0,1,0]])
        y_pred = K.constant([[0.22,0.13,0.65], [0.1,0.9,0]])
    """

    # TRANSFORM y_pred TO ONE-HOT
    # Get index of maximum
    max_inds = K.argmax(y_pred, axis=1)
    # Create an array of column indices in each row
    inds = K.arange(0, K.int_shape(y_pred)[1], dtype=max_inds.dtype)[None, :]
    # Create boolean mask of maximums
    bmask = K.equal(inds, max_inds[:, None])
    # Convert boolean mask to ones and zeros
    imask = K.cast(bmask, dtype='float32')

    y_true_t, y_pred_t = K.transpose(y_true), K.transpose(imask)

    true_positives = K.sum(K.round(K.clip(y_true_t * y_pred_t, 0, 1)), axis=1)
    possible_positives = K.sum(K.round(K.clip(y_true_t, 0, 1)), axis=1)
    predicted_positives = K.sum(K.round(K.clip(y_pred_t, 0, 1)), axis=1)

    precision = K.sum(true_positives) / (K.sum(predicted_positives) + K.epsilon())
    recall = K.sum(true_positives) / (K.sum(possible_positives)  + K.epsilon())

    return 2*((precision*recall)/(precision+recall))

def f1_macro(y_true, y_pred):
    """ Calculate Macro average F1-score for multi-class classification 
    Exemple : 
        y_true = K.constant([[0,0,1], [0,1,0]])
        y_pred = K.constant([[0.22,0.13,0.65], [0.1,0.9,0]])
    """
    # TRANSFORM y_pred TO ONE-HOT
    # Get index of maximum
    max_inds = K.argmax(y_pred, axis=1)
    # Create an array of column indices in each row
    inds = K.arange(0, K.int_shape(y_pred)[1], dtype=max_inds.dtype)[None, :]
    # Create boolean mask of maximums
    bmask = K.equal(inds, max_inds[:, None])
    # Convert boolean mask to ones and zeros
    imask = K.cast(bmask, dtype='float32')

    y_true_t, y_pred_t = K.transpose(y_true), K.transpose(imask)

    true_positives = K.sum(K.round(K.clip(y_true_t * y_pred_t, 0, 1)), axis=1)
    possible_positives = K.sum(K.round(K.clip(y_true_t, 0, 1)), axis=1)
    predicted_positives = K.sum(K.round(K.clip(y_pred_t, 0, 1)), axis=1)

    precision = K.mean(Lambda(lambda tensors: tensors[0] / (tensors[1]+ K.epsilon()))([true_positives, predicted_positives]))
    recall = K.mean(Lambda(lambda tensors: tensors[0] / (tensors[1]+ K.epsilon()))([true_positives, possible_positives])) 

    return 2*((precision*recall)/(precision+recall))