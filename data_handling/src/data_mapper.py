"""
Written by Robert Baumgartner, 2024
r.baumgartner-1@tudelft.nl
"""

import math
import copy
import numpy as np

from .configuration import InputConfig

class _SlidingWindow():
  """
  This class models a sliding window. Along the mapped features it also holds
  the IDs for each of the flows.
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

    self.N = 0
    self.LS = [0] * _SlidingWindow.n_features
    self.SS = [0] * _SlidingWindow.n_features


  def insert_flow(self, features, flow_id, label=None):
    """Inserts a flow into the window. 

    Args:
        features (list): The selected features as a list.
        flow_id (int): ID of the flow.
        label (int): The label. 0 for benign, 1 for malign. If no label is provided it is None. Only the seq channels tracks labels.
    """
    self.N += 1
    mapped_features = list()
    for idx, f in enumerate(features):
      self.LS[idx] += f
      self.SS[idx] += f**2

      mean = self.LS[idx] / self.N
      mapped_features.append(mean)
      std_dev = math.sqrt( abs( (self.SS[idx]/2) - mean**2) )
      mapped_features.append(std_dev)

    self.window.append(np.array(mapped_features))
    self.ids.append(flow_id)
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


class DataMapper():
  def __init__(self, data_path: str, settings_path: str, window_size: int):
    """
    Responsible for creating IDs, extracting the features from the flows, holding the configurations, and mapping to the
    respective channels. Does NOT compute the statistics, that is for the sliding window to do.
    """
    self.log_transform = True # TODO: perhaps we just do this flag away? No need to not do this

    self.data_path = data_path
    self.configs = InputConfig()
    self.configs.read_settings(settings_path)
    print("Read the input data setting with the following configurations: \n", self.configs)

    # these are the data structures mapping the ids to the windows
    self.current_idx = 0
    self.seq_store = _SlidingWindow(window_size=window_size, n_features=len(self.configs.feature_map), store_labels=True)
    self.src_store = dict() # #src ip to window
    self.dst_store = dict() # dst ip to window
    self.con_store = dict() # src_ip + dst_ip to window

    self.input_fh = open(self.data_path, "rt")
    if self.configs.has_header:
      self.input_fh.readline() # we do away with the header

  def __del__(self):
    self.input_fh.close()

  def get_next_window(self):
    """
    Reads a flow, maps it to the different channels, and returns a concatenated window 
    along with the mapped label. Label = 0 for benign, 1 for malicious.
    """
    line = self.input_fh.readline()
    if line=="":
      return None # reached eof
    
    flow = line.strip().split(self.configs.delimiter)
    label = 0 if flow[self.configs.label_idx] == self.configs.background_label else 1
    
    self._store_flow(flow, label)

    src_ip = flow[self.configs.src_ip]
    dst_ip = flow[self.configs.dst_ip]

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
    return _SlidingWindow.window_size, _SlidingWindow.n_features*2*4 # *2 for statistics, and *4 for 4 channels
  

  def _map_connection(self, src, dst):
    """A minor way to unify the formation of connection keys.

    Args:
        src (str): The respective IP address.
        dst (str): The respective IP address.
    """
    return src + "_" + dst

  def _insert_into_store(self, store, key, features):
    """Generic function to insert into the store (src, dst, con). Avoids boilerplate.

    Note: These ones do not use labels, therefore we don't need a label interface here.

    Args:
        store (dict): src_store, ...
        key (str): The IP address or connection.
        features (list): The features.
    """
    if key in store:
      store[key].insert_flow(features, self.current_idx)
    else:
      store[key] = _SlidingWindow()
      store[key].insert_flow(features, self.current_idx)

  def _store_flow(self, flow, label):
    """Maps the flow to the channels, and updates the internal datastructures accordingly.

    Args:
        flow (list): The flow, a list of features. [Does not have to be a flow necessarily, but this is what we used in our experiments].
        flow_id (int): The id of the flow.
    """
    features = list()
    for idx, ftype in self.configs.feature_map.items():
      if ftype==0 and self.log_transform:
        features.append(math.log(float(flow[idx])+1))
      elif ftype==0:
        features.append(float(flow[idx])+1)
      else:
        raise ValueError("Categorical features not supported yet.")
      
    src_ip = flow[self.configs.src_ip]
    dst_ip = flow[self.configs.dst_ip]

    self.seq_store.insert_flow(features, self.current_idx, label)
    self._insert_into_store(self.src_store, src_ip, features)
    self._insert_into_store(self.dst_store, dst_ip, features)
    self._insert_into_store(self.con_store, self._map_connection(src_ip, dst_ip), features)
    self.current_idx += 1