"""
Written by Robert Baumgartner, 2024
r.baumgartner-1@tudelft.nl
"""

import numpy as np

class ModelTrainer():
  """
  Object that trains a given model.
  """
  
  def __init__(self, data_handler, model_name: str, n_training_examples: int):
    self.data_handler = data_handler
    self.model_name = model_name
    self.n_training_examples = n_training_examples
    
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
      next_example, label = self.data_handler.get_next_window()
      #next_example = np.expand_dims(next_example, 0)
      if benign_training and label==1:
        continue
      train_batch.append(next_example)
      self.count += 1

      if i % int(1e5)==0:
        print("{} out of {} examples checked.".format(i, self.n_training_examples))
    
    train_batch = np.array(train_batch)
    model.fit(
      train_batch, 
      train_batch.reshape(train_batch.shape[0], -1),
      batch_size=b_size,
      epochs=1,
      shuffle=False
    )