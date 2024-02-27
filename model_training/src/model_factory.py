"""
Written by Robert Baumgartner, 2024
r.baumgartner-1@tudelft.nl
"""

import os

from .model_definitions import VanillaAeWrapper, LstmAeWrapper

class ModelFactory():
  def __init__(self, model_name: str):
    self.model_name = model_name

  def get_model(self, trained_model: str, scaler, **kwargs):
    """Constructs and returns the model. 

    Args:
        trained_model (str): Full path to a trained model. Will be loaded if trained_model not None.
        kwargs (dict): Model specific parameters.
    """
    model = None
    if self.model_name == "vanilla_ae":
      model = VanillaAeWrapper(scaler, **kwargs)
    elif self.model_name == "lstm_ae":
      model = LstmAeWrapper(scaler, **kwargs)
    else:
      raise ValueError("Invalid model name in ModelFactory: {}".format(self.model_name))
    
    if trained_model:
      model.load_model(trained_model)
    return model