"""
Written by Robert Baumgartner, 2024
r.baumgartner-1@tudelft.nl
"""

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc, roc_auc_score

class ModelEvaluator():
  def __init__(self, model_name: str, data_handler):
    self.data_handler = data_handler
    self.model_name = model_name
    if self.model_name != "vanilla_ae":
      raise ValueError("Model name in evaluator not supported: {}".format(self.model_name))

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

      if max_eval_samples is not None and self.count == max_eval_samples:
        print("{}/{} samples evaluated.".format(self.count, max_eval_samples))
        break
      elif max_eval_samples is not None:
        print("{}/{} samples evaluated.".format(self.count, max_eval_samples))
      else:
        print("{} batches and {} evaluated".format(self.bcount, self.count))

    print("Done with evaluation. Printing results.")

    fpr, tpr, thresholds = roc_curve(self.labels, self.predictions)
    roc_auc = roc_auc_score(self.labels, self.predictions)

    plt.figure(figsize=(10, 7))
    lw = 2
    sns.lineplot(
      fpr,
      tpr,
      color="darkorange",
      lw=lw,
      label="ROC curve (area = %0.2f)" % roc_auc,
    )
    sns.lineplot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig("aucs_vanilla_vae.png")

  def _collect_batch(self, bsize: int, max_eval_samples: int):
    """Collects a batch of size <= bsize and returns it as a list along with corresponding labels.

    Args:
        bsize (int): The back size.
    """
    batch = list()
    batch_labels = list()

    for _ in range(bsize):
      window_test = self.data_handler.get_next_window()
      if window_test is None: # reached eof
        return batch, batch_labels

      window, label = window_test[0], window_test[1]
      batch.append(window)
      batch_labels.append(label)

      self.count += 1
      if self.count == max_eval_samples:
        break

    return batch, batch_labels

  def _predict_vanilla_ae(self, model, batch):
    """
    What you think it does.

    TODO: shall we put this into the vanilla ae functions? This class could become a black hole
    """
    batch = np.array(batch)
    if batch.shape[0]==1: # batch is single window
      batch = np.expand_dims(batch, 0)
    res = model.predict(batch) #, verbose=0)
    diff = np.mean(np.abs(res - batch.reshape(batch.shape[0], -1)), axis=-1)
    return diff

  def _predict_next_batch(self, model, batch):
    """This function predicts the next value. Currently it returns us float values from the autoencoder, so
    that the RoC curves can be computed.
    """
    if self.model_name == "vanilla_ae":
      return self._predict_vanilla_ae(model, batch)