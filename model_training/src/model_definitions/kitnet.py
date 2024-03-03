"""
Written by Robert Baumgartner, 2024
r.baumgartner-1@tudelft.nl
"""

import numpy as np

from .base_class  import ModelWrapperBase
from .KitNET import KitNET

class KitNETWrapper(ModelWrapperBase):
  """The model wrapper class. Apart from training and predicting, this class is also 
  responsible for getting its own data right, e.g. reshaping the data and scaling it. 
  This provides a neat and clean interface for all classes using the model. 
  """
  def __init__(self, **kwargs):
    """
    KitNET has implicit scaling included already, therefore we don't need it.
    """
    self.n_train = kwargs["n_train"]
    input_shape = kwargs["input_shape"]
    input_dim = input_shape if type(input_shape)==int else input_shape[0]*input_shape[1] # whether input type is feature vector or n-gram

    FM_grace_period = min(5000, int(self.n_train/5))
    AD_grace_period = self.n_train - FM_grace_period
    self.model = self.AnomDetector = KitNET(input_dim, FM_grace_period=FM_grace_period, AD_grace_period=AD_grace_period)

    self.n_evaluated = 0

  def load_model(self, model_path: str):
    raise NotImplementedError("KitNET cannot be loaded or saved at the moment.")

  def save_model(self, model_path: str):
    print("WARNING: KitNET cannot be loaded or saved at the moment. Ignoring save.")

  def fit(self, X: list, **kwargs):
    """
    X is a list so the model can decide how it wants the data to be.
    """
    for i, x in enumerate(X):
      x = self._flatten_x(x)
      self.model.process(x)
      if i%1000==0:
        print("{} out of {} samples trained.".format(i, self.n_train))

  def predict(self, X: list):
    """
    X is a list so the model can decide how it wants the data to be.
    """
    res = list()
    for x in X:
      x = self._flatten_x(x)
      res.append(self.model.process(x))

      self.n_evaluated += 1
      if self.n_evaluated % 1000 == 0:
        print("{} samples evaluated.".format(self.n_evaluated)) # TODO: what's this about?

    return res
  
  def _flatten_x(self, x):
    return x.reshape(-1) 