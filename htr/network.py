import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, Dense
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.layers import MaxPooling2D, Input, Reshape

from tensorflow.keras.utils import Progbar


def ctc_loss(y_true, y_pred):
    """Function for computing the CTC loss"""

    if len(y_true.shape) > 2:
        y_true = tf.squeeze(y_true)

    '''
    y_pred.shape = (batch_size, string_length, alphabet_size_1_hot_encoded)
    
              Output layer of the model is softmax. So sum across alphabet_size_1_hot_encoded results 1.
              string_length give string length.
    '''
    input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
    input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)

    # y_true strings are padded with 0. So sum of non-zero gives number of characters in this string.
    label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64")

    '''
    About K.ctc_batch_loss:
        https://docs.w3cub.com/tensorflow~python/tf/keras/backend/ctc_batch_cost
        https://stackoverflow.com/questions/60782077/how-do-you-use-tensorflow-ctc-batch-cost-function-with-keras
    '''
    loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    # average loss across all entries in the batch
    loss = tf.reduce_mean(loss)

    return loss


def ctc_decode(Y_pred, batch_size=32, greedy=False, beam_width=10, top_paths=1, verbose=0):
    '''
    Use beam/greedy search to find most probable prediction.
    '''
    if verbose == 1:
        progbar = Progbar(target=np.ceil(len(Y_pred) / batch_size).astype(int))

    preds = []
    probs = []
    for i, batch in enumerate(range(0, len(Y_pred), batch_size)):
        y_pred = Y_pred[batch: batch + batch_size]
        decoded = K.ctc_decode(y_pred=y_pred,
                               input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1],
                               greedy=greedy,
                               beam_width=beam_width,
                               top_paths=top_paths)

        preds.extend([[c for c in y if c != -1] for y in decoded[0][0]])
        probs.extend(np.exp(decoded[1]).flatten())

        if verbose == 1:
            progbar.update(i+1)

    return preds, probs


def get_callbacks(logdir, checkpoint_filepath, stop_patience, reduce_patience, monitor="val_loss", verbose=0):
    '''
    Setup the list of callbacks for the model.
    '''
    return [
        CSVLogger(filename=os.path.join(logdir, 'epochs.log'),
                  separator=',',
                  append=True),

        TensorBoard(log_dir=logdir,
                    histogram_freq=3,
                    profile_batch=0,
                    update_freq='epoch'),

        ModelCheckpoint(filepath=checkpoint_filepath,
                        monitor=monitor,
                        save_best_only=True,
                        save_weights_only=True,
                        verbose=verbose),

        EarlyStopping(monitor=monitor,
                      min_delta=1e-8,
                      patience=stop_patience,
                      restore_best_weights=True,
                      verbose=verbose),

        ReduceLROnPlateau(monitor=monitor,
                          min_delta=1e-8,
                          factor=0.2,
                          patience=reduce_patience,
                          verbose=verbose)
    ]


def puigcerver(input_size, d_model):
    '''
    Convolucional Recurrent Neural Network by Puigcerver et al.
    Reference:
        Joan Puigcerver.
        Are multidimensional recurrent layers really necessary for handwritten text recognition?
        In: Document Analysis and Recognition (ICDAR), 2017 14th
        IAPR International Conference on, vol. 1, pp. 67–72. IEEE (2017)
        Carlos Mocholí Calvo and Enrique Vidal Ruiz.
        Development and experimentation of a deep learning system for convolutional and recurrent neural networks
        Escola Tècnica Superior d’Enginyeria Informàtica, Universitat Politècnica de València, 2018
    '''

    input_data = Input(name="input", shape=input_size)

    cnn = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same")(input_data)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=80, kernel_size=(3, 3), strides=(1, 1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)

    shape = cnn.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)

    blstm = Dropout(rate=0.5)(blstm)
    output_data = Dense(units=d_model, activation="softmax")(blstm)

    return (input_data, output_data)
