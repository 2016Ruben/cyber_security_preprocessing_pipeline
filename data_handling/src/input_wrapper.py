"""
Written by Robert Baumgartner, 2024
r.baumgartner-1@tudelft.nl
"""

import sys
from datetime import datetime

import math

from .configuration import FileInputConfigs, StreamInputConfigs

_kitsune_path = "../../kitsune/Kitsune"
try:
  sys.path.append(_kitsune_path)
  from FeatureExtractor import FE
except ImportError:
  print("FeatureExtractor.FE from Kitsune not found in directory\
        {}. Continuing without it.".format(_kitsune_path))
  sys.path.remove(_kitsune_path)
del _kitsune_path

class InputWrapper():
  """
  This class abstracts away the input for us and provides a common interface for all possible input we might face.
  """
  def __init__(self, data_path: str, settings_path: str, labelf_path: str, input_type: str, use_timediff: bool):
    self.log_transform = True # TODO: perhaps we just do this flag away? No need to not do this
    self.use_timediff = use_timediff
    
    # prepare the configurations
    self.input_type = input_type
    if input_type == "csv":
      self.configs = FileInputConfigs()
    elif input_type == "kitsune":
      self.configs = StreamInputConfigs()
    else:
      raise ValueError("Unknown --input_type argument {}. Only 'csv' or 'kitsune' supported so far.".format(input_type))
    self.configs.read_settings(settings_path)
    print("Read the input data setting with the following configurations: \n", self.configs)

    # open the handlers
    self.data_path = data_path
    self.input_handler = None
    if input_type == "csv":
      self.input_handler = open(self.data_path, "rt")
      if self.configs.has_header:
        self.input_handler.readline() # we do away with the header
    elif input_type == "kitsune":
      self.input_handler = FE(data_path)

    self.labelf_path = labelf_path
    if self.labelf_path is not None:
      self.labelf_handler = open(self.labelf_path, "rt")
    if self.labelf_path is not None and self.configs.labelf_has_header:
      self.labelf_handler.readline() # throw away that header

  def __del__(self):
    print("Closing input files.")
    if self.input_type == "csv":
      self.input_handler.close()
    if self.labelf_path is not None:
      self.labelf_handler.close()

  def n_features(self):
    """Gets the number of raw features (without temporal features)
    to be extracted in each turn before processing the features further.
    """
    return len(self.configs.feature_map)

  def extract_features(self):
    """
    Gets the next feature vector, label, IP-addresses and the timestamp as a 
    floating point number. If no timestamp is provided it will be None. Feature 
    vector will be None if end of input is reached.
    """
    if self.input_type == "csv":
      all_features, label = self._process_line()
    elif self.input_type == "kitsune":
      all_features, label = self._get_streamed_line()

    if all_features is None:
      return None

    src_ip = all_features[self.configs.src_ip]
    dst_ip = all_features[self.configs.dst_ip]

    timestamp = None
    if self.use_timediff and len(self.configs.timestamp_format) > 0:
      timestamp = all_features[self.configs.timestamp_idx]
      dt_obj = datetime.strptime(timestamp, self.configs.timestamp_format)
      timestamp = dt_obj.timestamp()
    elif self.use_timediff: # already coming as float
      timestamp = all_features[self.configs.timestamp_idx]

    feature_vector = list()
    for idx, ftype in self.configs.feature_map.items():
      if ftype==0 and self.log_transform:
        feature_vector.append(math.log(float(all_features[idx])+1))
      elif ftype==0:
        feature_vector.append(float(all_features[idx])+1)
      else:
        raise ValueError("Categorical features not supported yet.")

    return feature_vector, label, src_ip, dst_ip, timestamp

  def _process_line(self):
    """
    Processes the next line of csv file and returns all the features
    without selecting the proper ones, as well as the label.

    The return can be None. In this case the input file has reached eof.
    """
    line = self.input_handler.readline()
    if line=="":
      return None, None # reached eof
    
    line = line.strip().split(self.configs.delimiter)
    if self.labelf_path is not None:
      label = self._get_label_from_file()
    else:
      label = self._label_from_string(line[self.configs.label_idx])
    
    return line, label
  
  def _get_streamed_line(self):
    """
    Gets the next streamed line.

    The return can be None. In this case the input file has reached eof.
    """
    features = self.input_handler.get_next_vector_unprocessed()
    if len(features) == 0:
      return None, None # reached eof
    
    if self.labelf_path is not None:
      label = self._get_label_from_file()
    else:
      label = self._label_from_string(features[self.configs.label_idx])
    return features, label

  def _label_from_string(self, string_label: str):
    """Label string to integer. Does so by comparing with the background label
    as set by the yaml-settings file.

    Args:
        string_label (str): The label as string as appears in dataset.

    Returns:
        int: 0 for benign, 1 for malign.
    """
    return 0 if string_label == self.configs.background_label else 1

  def _get_label_from_file(self):
    """
    Does what you think it does.
    """
    line = self.labelf_handler.readline()
    line = line.split(self.configs.labelf_delimiter)
    return self._label_from_string(line[self.configs.label_idx])