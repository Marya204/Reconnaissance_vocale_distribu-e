import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Dropout, Bidirectional, LSTM, TimeDistributed, BatchNormalization, Lambda

def reshape_layer(t):
    return tf.reshape(t, (tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2]*tf.shape(t)[3]))

def build_model(input_shape=(128, 128, 1), num_classes=len("abcdefghijklmnopqrstuvwxyz ") + 1):
    inputs = Input(name='input', shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)
    x = BatchNormalization()(x)

    x = Lambda(reshape_layer)(x)  # <- utiliser la fonction nommée

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.25)(x)

    x = TimeDistributed(Dense(num_classes, activation='softmax'))(x)

    model = Model(inputs=inputs, outputs=x, name='ASR_Model')
    return model
