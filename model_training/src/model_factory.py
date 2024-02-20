from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Reshape
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional
from tensorflow.keras.layers import Lambda, Flatten
import keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping


class ModelFactory():
  def __init__(self):
    pass

  def get_vanilla_ae(self, **kwargs):
    """
    Returns a vanilla autoencoder model.

    TODO: you can potentially set the other parameters as well by kwargs. Does that make sense?
    """
    input_shape = kwargs.get("input_shape")
    output_dim = None # TODO: derive from the input shape

    input_layer = Input(shape=input_shape)
    encoder = Dense(int(output_dim * 0.5), activation="relu")
    x_encoder = encoder(input_layer)
    encoder_output = Dense(int(output_dim * 0.25), activation="relu")(x_encoder)

    x_latent = Dense(int(output_dim * 0.5), activation="relu")(encoder_output)
    x_out = Dense(output_dim, activation="linear")(x_latent)
    output = x_out

    model = Model(input_layer, output)
    model.compile(
      loss="mean_absolute_error",
      optimizer="adam",
      metrics=["mean_absolute_error"]
    )
    return model


  def get_model(self, model_name, **kwargs):
    """Constructs and returns the model. 

    Args:
        model_name (string): Currently only supports 'vanilla_ae'.
    """
    if model_name == 'vanilla_ae':
      return self.get_vanilla_ae(**kwargs)
    else:
      raise ValueError("Invalid model name in ModelFactory: {}".format(model_name))