########################################################################
# import python-library
########################################################################
# from import
import keras.models
from keras import backend as K
from keras.layers import Input, Dense, BatchNormalization, Activation, LayerNormalization
from keras.models import Model
import tensorflow as tf


########################################################################
# keras model
########################################################################
def denseNN(input_dim,output_dim,lr,nbLayers=1,normalization="layer",activation="gelu"):
    """
    define the keras model
    the model based on AST MLP head (at least with the default parameters)
    """

    x = Input(shape=(input_dim,))

    for i in range(nbLayers):
        h = Dense(output_dim)(x)
        if normalization=="layer":
            h = LayerNormalization() (h)
        elif normalization=="batch":
            h = BatchNormalization()(h)

        if activation=="gelu":
            h = Activation("gelu")(h)
        elif activation=="relu":
            h = Activation('relu')(h)

    model = Model(inputs=x, outputs=h)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                  loss='mean_squared_error')

    return model


#########################################################################

def load_model(file_path):
    return keras.models.load_model(file_path, compile=False)


def clear_session():
    K.clear_session()
