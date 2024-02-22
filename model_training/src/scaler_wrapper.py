"""
Written by Robert Baumgartner, 2024
r.baumgartner-1@tudelft.nl
"""

import os
import pickle as pk

import numpy as np

from sklearn.preprocessing import MinMaxScaler

class ScalerWrapper():
  """
  A wrapper class for the MinMaxScaler. The MinMaxScaler does not support 3D arrays, 
  and this class helps with convecience functionality for that.
  """
  def __init__(self):
    self.scaler = MinMaxScaler()

  def fit_transform_3d(self, X):
    X, X_shape = self._reshape_3d_to_2d(X)
    self.scaler.fit_transform(X)
    return self._reshape_2d_to_3d(X, X_shape)

  def transform_3d(self, X):
    X, X_shape = self._reshape_3d_to_2d(X)
    self.scaler.transform(X)
    return self._reshape_2d_to_3d(X, X_shape)
  
  def fit_transform_2d(self, X):
    raise Exception("Not implemented yet.")

  def transform_2d(self, X):
    raise Exception("Not implemented yet.")

  def save_state(self, scaler_save_path):
    """Saves the stored scaler and creates the directory structure to it if it doesn't exist yet.

    Args:
        scaler_save_path (str): Full path to scaler.
    """
    path_split = os.path.split(scaler_save_path)
    directory = os.path.join(*(path_split[:-1]))
    print("Storing model in {}".format(scaler_save_path))
    if not os.path.isdir(directory):
      os.mkdir(directory)

    pk.dump(self.scaler, open(scaler_save_path, "wb"))

  def load_state(self, scaler_save_path):
    """Loads the scaler into the wrapper.

    Args:
        scaler_save_path (str): Full path to the scaler.

    Raises:
        Exception: File does not exist.
    """
    if not os.path.isfile(scaler_save_path):
      raise Exception("Scaler does not exist. Adjust scaler save path: {}".format(scaler_save_path))
    self.scaler = pk.load(open(scaler_save_path, "rb"))

  def _reshape_3d_to_2d(self, X):
    """Reshapes and returns reshaped array as well as original shape.

    Args:
        X (np.array): The 3D array.
    """
    X_shape = X.shape
    return X.reshape(X_shape[0], -1), X_shape

  def _reshape_2d_to_3d(self, X, shape_3d):
    """Reverses the 3D to 2D conversion done by _reshape_3d_to_2d.

    Args:
        X (np.array): 2d array
        shape_3d (np.array): The 3D shape we want to reshape into (returned from e.g. _reshape_3d_to_2d)
    """
    return X.reshape(*list(shape_3d))
