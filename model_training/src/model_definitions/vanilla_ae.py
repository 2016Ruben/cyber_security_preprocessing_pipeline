"""
Written by Robert Baumgartner, 2024
r.baumgartner-1@tudelft.nl
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import LSTM, Bidirectional

from tensorflow.keras.models import load_model

def load_vanilla_ae(trained_model_path: str):
  """
  No need to explain here.
  """
  return load_model(trained_model_path)

def get_uncompiled_vanilla_ae(input_shape, output_dim):
  """
  Get the vanilla autoencoder.

  Returns:
      tf.Model: The uncompiled model.
  """
  input_layer = Input(shape=input_shape)
  encoder = LSTM(int(output_dim * 0.5), activation="relu")
  x_encoder = encoder(input_layer)
  encoder_output = Dense(int(output_dim * 0.25), activation="relu")(x_encoder)

  x_latent = Dense(int(output_dim * 0.5), activation="relu")(encoder_output)
  x_out = Dense(output_dim, activation="linear")(x_latent)
  output = x_out

  return Model(input_layer, output)