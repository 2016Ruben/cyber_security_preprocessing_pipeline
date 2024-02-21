"""
Written by Robert Baumgartner, 2024
r.baumgartner-1@tudelft.nl
"""

from .model_definitions import get_uncompiled_vanilla_ae

class ModelFactory():
  def __init__(self, model_name: str):
    self.model_name = model_name

  def get_model(self, **kwargs):
    """Constructs and returns the model. 

    Args:
        model_name (string): Currently only supports 'vanilla_ae'.
    """
    if self.model_name == 'vanilla_ae':
      return self._get_vanilla_ae(**kwargs)
    else:
      raise ValueError("Invalid model name in ModelFactory: {}".format(self.model_name))
    
  def _get_vanilla_ae(self, **kwargs):
    """
    Returns a vanilla autoencoder model.

    TODO: you can potentially set the other parameters as well by kwargs. Does that make sense?
    """
    input_shape = kwargs["input_shape"]
    output_dim = input_shape[0] * input_shape[1]

    model = get_uncompiled_vanilla_ae(input_shape=input_shape, output_dim=output_dim)
    model.compile(
      loss="mean_absolute_error",
      optimizer="adam",
      metrics=["mean_absolute_error"]
    )
    return model