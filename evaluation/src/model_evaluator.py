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
    
  def _predict_next_value(self, model, window):
    """This function predicts the next value. Currently it returns us float values from the autoencoder, so
    that the RoC curves can be computed.
    """
    if self.model_name != "vanilla_ae":
      res = model.predict(np.expand_dims(window, 0))
      diff = np.mean(np.abs(res - window.reshape(-1)))
      return diff
    
  def evaluate(self, model):
    values = list()
    labels = list()

    while True:
      window_test = self.data_handler.get_next_window()
      if window_test is None: # reached eof
        break
      
      window, label = window_test[0], window_test[1]
      next_value = self._predict_next_value(model, window)
      values.append(next_value)
      labels.append(label)

    fpr, tpr, thresholds = roc_curve(labels, values)
    roc_auc = roc_auc_score(labels, values)

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
    