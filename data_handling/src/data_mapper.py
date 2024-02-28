"""
Written by Robert Baumgartner, 2024
r.baumgartner-1@tudelft.nl
"""

import copy
import numpy as np
import math

from .input_wrapper import InputWrapper

class _SlidingWindow():
  """
  This class models a sliding window. Along the mapped features it also holds the IDs
  for each of the flows/vectors and is responsible for the correct computation of its features.
  """
  # static attributes
  window_size = 0
  n_features = 0

  def __init__(self, window_size: int=None, n_features: int=None, store_labels:bool=False):
    if window_size is not None:
      # we only need to initialize these two once
      assert(n_features is not None)
      _SlidingWindow.window_size = window_size
      _SlidingWindow.n_features = n_features

    self.window = list()
    self.ids = list()
    self.labels = list() if store_labels else None
    self.last_timestamp = 0

    self.N = 0
    self.LS = [0] * _SlidingWindow.n_features
    self.SS = [0] * _SlidingWindow.n_features

  def insert_feature_vector(self, features, vector_id:int, label:int=None, timestamp:float=None):
    """Inserts a feature_vector into the window. 

    Args:
        features (list): The selected features as a list.
        vector_id (int): ID of the feature_vector.
        label (int): The label. 0 for benign, 1 for malign. If no label is provided it is None. Only the seq channels tracks labels.
        timestamp (float): The timestamp in seconds. We use this one for the timediff-feature.
    """
    self.N += 1
    mapped_features = list()
    for idx, f in enumerate(features):
      self.LS[idx] += f
      self.SS[idx] += f**2

      mapped_features = self._insert_statistics(mapped_features, idx)

    if timestamp is not None:
      diff = timestamp - self.last_timestamp
      diff = math.log(diff + 1)
      self.LS[-1] = diff
      self.SS[-1] = diff**2
      mapped_features = self._insert_statistics(mapped_features, -1)
      self.last_timestamp = timestamp

    self.window.append(np.array(mapped_features))
    self.ids.append(vector_id)
    if label:
      self.labels.append(label)

    if len(self.ids) > _SlidingWindow.window_size:
      self.window.pop(0)
      self.ids.pop(0)
      if label:
        self.labels.pop(0)

  def retrieve_window(self):
    """Returns the window. Does zero padding if the window is smaller than window size.

    Returns:
        np.array: The (potentially padded) window.
    """
    if len(self.window)==_SlidingWindow.window_size:
      return self.window
    
    res = copy.copy(self.window)
    for _ in range(_SlidingWindow.window_size-len(self.window)):
      res.insert(0, np.zeros((_SlidingWindow.n_features*2))) # *2 because of mean and std-deviation
    return res

  def _insert_statistics(self, res_list, idx):
    """Computest the statistics from LS and SS and inserts them into list at the 
    given index. Returns updated list.

    Args:
        res_list (list): The list you want to append the features to.
        idx (int): The index of LS and SS.
    """
    mean = self.LS[idx] / self.N
    res_list.append(mean)
    std_dev = math.sqrt( abs( (self.SS[idx]/2) - mean**2) )
    res_list.append(std_dev)
    return res_list


class DataHandler():
  def __init__(self, data_path: str, settings_path: str, labelf_path: str, input_type: str, window_size: int, use_timediff: bool):
    """
    Responsible for creating IDs, extracting the features from the flows/vectors, holding the configurations, and mapping to the
    respective channels. Does NOT compute the statistics, that is for the sliding window to do.
    """
    self.input_handler = InputWrapper(data_path, settings_path, labelf_path, input_type, use_timediff)
    self.use_timediff = use_timediff
    self.input_type = input_type

    n_features = self.input_handler.n_features()
    if self.use_timediff:
      n_features += 1

    # these are the data structures mapping the ids to the windows
    self.current_idx = 0
    self.seq_store = _SlidingWindow(window_size=window_size, n_features=n_features, store_labels=True)
    self.src_store = dict() # #src ip to window
    self.dst_store = dict() # dst ip to window
    self.con_store = dict() # src_ip + dst_ip to window


  def get_next_window(self):
    """
    Reads a feature_vector, maps it to the different channels, and returns a concatenated window 
    along with the mapped label. Label = 0 for benign, 1 for malicious. Returns None if end of input
    reached.
    """

    features, label, src_ip, dst_ip, timestamp = self.input_handler.extract_features()
    if features is None:
      return None, None
    elif self.input_type == "kitsune_original":
      return features, label

    self._store_feature_vector(features, label, src_ip, dst_ip, timestamp)

    seq_ngram = np.array(self.seq_store.retrieve_window())
    src_ngram = np.array(self.src_store[src_ip].retrieve_window())
    dst_ngram = np.array(self.dst_store[dst_ip].retrieve_window())
    con_ngram = np.array(self.con_store[self._map_connection(src_ip, dst_ip)].retrieve_window())

    return np.hstack((seq_ngram, src_ngram, dst_ngram, con_ngram)), label
  
  def get_input_shape(self):
    """
    Gets the input shape that's to be expected for the model. The shape is inferred from the features as declared in the 
    settings, as well as from the window_size.
    """
    if self.input_type == "kitsune_original":
      return self.input_handler.n_features()
    return _SlidingWindow.window_size, _SlidingWindow.n_features*2*4 # *2 for statistics, and *4 for 4 channels
  

  def _map_connection(self, src, dst):
    """A minor way to unify the formation of connection keys.

    Args:
        src (str): The respective IP address.
        dst (str): The respective IP address.
    """
    return src + "_" + dst

  def _insert_into_store(self, store, key, features, timestamp: float):
    """Generic function to insert into the store (src, dst, con). Avoids boilerplate.

    Note: These ones do not use labels, therefore we don't need a label interface here.

    Args:
        store (dict): src_store, ...
        key (str): The IP address or connection.
        features (list): The features.
    """
    if key in store:
      store[key].insert_feature_vector(features, self.current_idx, timestamp=timestamp)
    else:
      store[key] = _SlidingWindow()
      store[key].insert_feature_vector(features, self.current_idx, timestamp=timestamp)

  def _store_feature_vector(self, feature_vector: list, label: int, src_ip: str, dst_ip: str, timestamp: float):
    """Maps the feature_vector along with the provided features to the channels, and updates the internal datastructures accordingly.
    """
    self.seq_store.insert_feature_vector(feature_vector, self.current_idx, label, timestamp=timestamp)
    self._insert_into_store(self.src_store, src_ip, feature_vector, timestamp)
    self._insert_into_store(self.dst_store, dst_ip, feature_vector, timestamp)
    self._insert_into_store(self.con_store, self._map_connection(src_ip, dst_ip), feature_vector, timestamp)
    self.current_idx += 1