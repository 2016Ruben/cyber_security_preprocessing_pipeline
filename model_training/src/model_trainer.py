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
  
  def __init__(self, data_handler, scaler, model_name: str, n_training_examples: int, save_model: bool, model_save_path: str):
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
    self.scaler = scaler

    self.model_name = model_name
    self.n_training_examples = n_training_examples

    self.save_model = save_model
    self.model_save_path = model_save_path
    
    self.count = 0

  def train(self, model, benign_training: bool, **kwargs):
    if self.model_name == "vanilla_ae":
      b_size = kwargs["b_size"]
      self._train_tf_model_batch(model, benign_training, b_size)
    else:
      raise ValueError("Model type not supported in ModelTrainer: {}".format(self.model_name))
    
    print("Trained model with {} training examples".format(self.count))
    
  def _train_tf_model_batch(self, model, benign_training: bool, b_size: int):
    """Although the data handler supports streaming by returning a single instance each time, 
    the tensforflow API does better for us just simply defining one large array and learning
    for one epoch unshuffled. The effect is the same.
    """
    train_batch = list()
    for i in range(self.n_training_examples):
      res = self.data_handler.get_next_window()
      if res is None:
        print("EOF reached during training phase. Please choose a larger input file or adjust\
               'n_training_examples' parameter. Terminating program.")
        exit()

      next_example, label = res[0], res[1]
      if benign_training and label==1:
        continue
      train_batch.append(next_example)
      self.count += 1

      if i % int(1e5)==0:
        print("{} out of {} examples checked.".format(i, self.n_training_examples))
    
    train_batch = np.array(train_batch)
    if self.scaler is not None:
      train_batch = self.scaler.fit_transform_3d(train_batch)

    model.fit(
      train_batch, 
      train_batch.reshape(train_batch.shape[0], -1),
      batch_size=b_size,
      epochs=1,
      shuffle=False
    )

    if self.save_model:
      path_split = os.path.split(self.model_save_path)
      directory = os.path.join(*(path_split[:-1]))
      print("Storing model in {}".format(self.model_save_path))
      if not os.path.isdir(directory):
        os.mkdir(directory)

      model.save(self.model_save_path)