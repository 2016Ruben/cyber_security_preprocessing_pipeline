"""
Written by Robert Baumgartner, 2024
r.baumgartner-1@tudelft.nl
"""

import os

import numpy as np

class ModelTrainer():
  """
  Object that trains a given model.
  """
  
  def __init__(self, data_handler, n_training_examples: int, save_model: bool, model_save_path: str):
    """Initializes the trainer.

    Args:
        data_handler (DataHandler): The wrapper for the input data.
        scaler (ScalerWrapper): The scaler wrapper to be used. If None no scaling will be applied.
        model_name (str): The name of the model type.
        n_training_examples (int): Number of samples to be used for training.
        save_model (bool): Save the model, yes or no?
        model_save_path (str): If model is to be save it'll be here.
    """
    self.data_handler = data_handler
    self.n_training_examples = n_training_examples

    self.save_model = save_model
    self.model_save_path = model_save_path
    
    self.count = 0

  def train(self, model, benign_training: bool, **kwargs):
    if "b_size" in kwargs:
      b_size = kwargs["b_size"]
      self._train_tf_model_batch(model, benign_training, b_size)
    else:
      raise NotImplementedError("So far b_size must be supported")
    
    print("Trained model with {} training examples".format(self.count))
    
  def _train_tf_model_batch(self, model, benign_training: bool, b_size: int):
    """Although the data handler supports streaming by returning a single instance each time, 
    the tensforflow API does better for us just simply defining one large array and learning
    for one epoch unshuffled. The effect is the same.
    """
    train_batch = list()
    for i in range(self.n_training_examples):
      next_example, label = self.data_handler.get_next_window()
      if next_example is None:
        print("EOF reached during training phase. Please choose a larger input file or adjust\
               'n_training_examples' parameter. Terminating program.")
        exit()

      if benign_training and label==1:
        continue
      train_batch.append(next_example)
      self.count += 1

      if i % int(1e5)==0:
        print("{} out of {} examples checked.".format(i, self.n_training_examples))
    
    if len(train_batch)==0:
      print("No viable training examples found. Is the labelfile correctly set, and are its settings correct?.")
      exit()

    model.fit(train_batch, b_size)

    if self.save_model:
      path_split = os.path.split(self.model_save_path)
      directory = os.path.join(*(path_split[:-1]))
      print("Storing model in {}".format(self.model_save_path))
      if not os.path.isdir(directory):
        os.mkdir(directory)

      model.save_model(self.model_save_path)