"""
Written by Robert Baumgartner, 2024
r.baumgartner-1@tudelft.nl
"""

import os

from .model_definitions import get_uncompiled_vanilla_ae, load_vanilla_ae

class ModelFactory():
  def __init__(self, model_name: str):
    self.model_name = model_name

  def get_model(self, trained_model: str, **kwargs):
    """Constructs and returns the model. 

    Args:
        trained_model (string): Path to a trained model. If no model to be loaded None.
        kwargs (dict): Model specific parameters.
    """
    if self.model_name == "vanilla_ae":
      return self._get_vanilla_ae(trained_model, **kwargs)
    else:
      raise ValueError("Invalid model name in ModelFactory: {}".format(self.model_name))
    
  def _get_vanilla_ae(self, trained_model: str, **kwargs):
    """
    Returns a vanilla autoencoder model.

    TODO: you can potentially set the other parameters as well by kwargs. Does that make sense?
    """
    if trained_model is not None:
      return load_vanilla_ae(trained_model)
    
    input_shape = kwargs["input_shape"]
    output_dim = input_shape[0] * input_shape[1]

    model = get_uncompiled_vanilla_ae(input_shape=input_shape, output_dim=output_dim)
    model.compile(
      loss="mean_absolute_error",
      optimizer="adam",
      metrics=["mean_absolute_error"]
    )
    return model