"""
Written by Robert Baumgartner, 2024
r.baumgartner-1@tudelft.nl
"""

from abc import ABC, abstractmethod

class ModelWrapperBase(ABC):
  """The model wrapper class. Apart from training and predicting, this class is also 
  responsible for getting its own data right, e.g. reshaping the data and scaling it. 
  This provides a neat and clean interface for all classes using the model. 
  """
  def __init__(self, **kwargs):
    pass

  @abstractmethod
  def load_model(self, model_path: str):
    pass

  @abstractmethod
  def save_model(self, model_path: str):
    pass

  @abstractmethod
  def fit(self, X: list):
    """X is a list so the model can decide how it wants the data to be.
    """
    pass

  @abstractmethod
  def predict(self, X: list):
    pass

  @abstractmethod
  def _preprocess_batch(self, X: list):
    pass