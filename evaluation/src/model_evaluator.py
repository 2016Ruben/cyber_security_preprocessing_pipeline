"""
Written by Robert Baumgartner, 2024
r.baumgartner-1@tudelft.nl
"""

import os
import pickle as pk

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score

class ModelEvaluator():
  def __init__(self, data_handler, model_name):
    """Initialize the evaluation model.

    Args:
        data_handler (DataHandler): The data handler.
        scaler (_type_): The scaler wrapper. If None no scaling will be applied. Make sure it is trained by trainer or loaded!
        model_name (str): The type of the model.

    Raises:
        ValueError: Model type does not exist.
    """
    self.data_handler = data_handler
    self.model_name = model_name

    self.count = 0
    self.bcount = 0

    self.predictions = list()
    self.labels = list()

  def evaluate(self, model, max_eval_samples: int, b_size: int):
    while True:
      batch, batch_labels = self._collect_batch(b_size, max_eval_samples)
      if len(batch)==0:
        break
      
      predictions = self._predict_next_batch(model, batch)
      self.predictions.extend(predictions)
      self.labels.extend(batch_labels)

      if len(batch)<b_size:
        break
      
      self.bcount += 1

      if max_eval_samples is not None and self.count >= max_eval_samples:
        print("{}/{} samples evaluated.".format(self.count, max_eval_samples))
        return
      elif max_eval_samples is not None:
        print("{}/{} samples evaluated.".format(self.count, max_eval_samples))
      else:
        print("{} batches and {} samples evaluated".format(self.bcount, self.count))

    print("{}/{} samples evaluated.".format(self.count, max_eval_samples))

  def print_plots(self, figure_path):
    """
    Prints the plots.
    """
    self._print_auc_real_time(figure_path)

  def save_results(self, save_path: str):
    """Saves the results as a dictionary in the specified path. 
    If directory does not exist it will be created.

    Args:
        save_path (str): The full path to save the model.
    """
    path_split = os.path.split(save_path)
    directory = os.path.join(*(path_split[:-1]))
    print("Storing results in {}".format(save_path))
    if not os.path.isdir(directory):
      os.mkdir(directory)
    
    results = {
      "predictions": self.predictions,
      "labels": self.labels
    }
    pk.dump(results, open(save_path, "wb"), protocol=-1)
    print("Saved results.")

  def _collect_batch(self, bsize: int, max_eval_samples: int):
    """Collects a batch of size <= bsize and returns it as a list along with corresponding labels.

    Args:
        bsize (int): The batch size.
    """
    batch = list()
    batch_labels = list()

    for _ in range(bsize):
      window, label = self.data_handler.get_next_window()
      if window is None: # reached eof
        return batch, batch_labels

      batch.append(window)
      batch_labels.append(label)

      self.count += 1
      if self.count == max_eval_samples:
        break

    return batch, batch_labels

  def _predict_tf_model(self, model, batch):
    """
    Predicts the tensorflow models. Unlike e.g. Kitsune they don't return the anomaly scores already, 
    so this function is an approach to do that in a unified manner.
    """
    res = model.predict(batch)
    batch = np.array(batch)

    # way 1
    #diff = np.mean(np.abs(res - batch.reshape(batch.shape[0], -1)), axis=-1)
    
    # way 2
    res = res.reshape(batch.shape)
    diff = np.mean(np.abs(res[:, -1, :] - batch[:, -1, :].reshape(batch.shape[0], -1)), axis=-1)
    return diff

  def _predict_next_batch(self, model, batch):
    """This function predicts the next value. Currently it returns us float values from the autoencoder, so
    that the RoC curves can be computed.
    """
    if self.model_name == "vanilla_ae" or self.model_name == "lstm_ae":
      return self._predict_tf_model(model, batch)
    elif self.model_name == "kitnet":
      return model.predict(batch)
    else:
      raise NotImplementedError("model_name is not implemented yet in _predict_next_batch")

  def _print_auc_real_time(self, figure_path):
    """Print the AuC of the real time mode: Only the last element is the labeling instance. 

    Args:
        figure_path (str): Path to where the figures will be stored. This is a path, not a file!
    """
    fpr, tpr, thresholds = roc_curve(self.labels, self.predictions)
    roc_auc = roc_auc_score(self.labels, self.predictions)

    plt.figure(figsize=(10, 7))
    lw = 2
    sns.lineplot(
      x=fpr,
      y=tpr,
      color="darkorange",
      lw=lw,
      label="ROC curve (area = %0.2f)" % roc_auc,
      errorbar=None, # to speed up computation, else very slow
    )

    outfile = None
    if not figure_path.endswith(".png") and not figure_path.endswith(".jpg"):
      outfile = os.path.join(figure_path, "auc_normal.png")
    else:
      outfile = figure_path
    print("Saving result figure in {}".format(outfile))

    sns.lineplot(x=[0, 1], y=[0, 1], color="navy", lw=lw, linestyle="--", errorbar=None,)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.savefig(outfile)