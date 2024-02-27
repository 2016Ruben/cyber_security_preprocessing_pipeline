"""
Written by Robert Baumgartner, 2024
r.baumgartner-1@tudelft.nl
"""

import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Bidirectional
from tensorflow.keras.models import load_model

from .base_class  import ModelWrapperBase

class LstmAeWrapper(ModelWrapperBase):
  """The model wrapper class. Apart from training and predicting, this class is also 
  responsible for getting its own data right, e.g. reshaping the data and scaling it. 
  This provides a neat and clean interface for all classes using the model. 
  """
  def __init__(self, scaler=None, **kwargs):
    input_shape = kwargs["input_shape"]
    self.model = self._init_model(input_shape)
    self.scaler = scaler

  def load_model(self, model_path: str):
    self.model = load_model(model_path)

  def save_model(self, model_path: str):
    self.model.save(model_path)

  def fit(self, X: list, b_size):
    """
    X is a list so the model can decide how it wants the data to be.
    """
    X = self._prepare_array(X)
    if self.scaler is not None:
      X = self.scaler.fit_transform_3d(X)

    self.model.fit(
      X, 
      X,
      batch_size=b_size,
      epochs=1,
      shuffle=False,
    )

  def predict(self, X: list):
    """
    X is a list so the model can decide how it wants the data to be.
    """
    X = self._prepare_array(X)
    if self.scaler is not None:
      X = self.scaler.transform_3d(X)
    return X
  
  def _prepare_array(self, X):
    n_samples = len(X)
    X = np.array(X)
    if n_samples==1:
      X = np.expand_dims(X, 0)
    return X

  def _init_model(self, input_shape):
    """
    Get the compiled LSTM autoencoder.
    """
    output_shape = input_shape[0] * input_shape[1]

    input_layer = Input(shape=input_shape)
    encoder = LSTM(int(output_shape * 0.5), activation="relu")
    x_encoder = encoder(input_layer)
    encoder_output = Dense(int(output_shape * 0.25), activation="relu")(x_encoder)

    x_latent = Dense(int(output_shape * 0.5), activation="relu")(encoder_output)
    x_out = Dense(output_shape, activation="linear")(x_latent)
    output = x_out

    model = Model(input_layer, output)

    model.compile(
      loss="mean_absolute_error",
      optimizer="sgd",
      metrics=["mean_absolute_error"]
    )

    return model