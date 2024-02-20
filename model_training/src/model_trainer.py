

class ModelTrainer():
  """
  Object that trains a given model.
  """
  
  def __init__(self):
    pass

  def train(self, model, model_type, **kwargs):
    if model_type == "vanilla_ae":
      self._train_vanilla_ae(model, model, **kwargs)
    else:
      raise ValueError("Model type not supported in ModelTrainer: {}".format(model_type))
    
  
  def _train_vanilla_ae(self, model, **kwargs):
    pass