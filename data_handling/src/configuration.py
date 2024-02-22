"""
Written by Robert Baumgartner, 2024
r.baumgartner-1@tudelft.nl
"""

import os
import yaml

class InputConfig():
  """
  For details, see ../readme.md
  """
  def __init__(self):

    self.__feature_type_mapping = {
      "continuous": 0,
      "categorical": 1
    }

    self.has_header = None
    self.delimiter = None
    
    self.src_ip = None
    self.dst_ip = None

    self.use_timediff = None
    self.timestamp_idx = None
    self.timestamp_format = None

    self.feature_map = dict() # mapping index to type. 0 for continuous, 1 for categorical

    self.label_idx = None
    self.background_label = None

  def read_settings(self, infile: str):
    if not os.path.isfile(infile):
      raise ValueError("Settings file does not exist: {}".format(infile))

    # for details on Loader in next line: https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
    yml_input = yaml.load(open(infile, "rt"), Loader=yaml.SafeLoader) 
    
    self.has_header = yml_input["structure"]["has_header"]
    self.delimiter = yml_input["structure"]["delimiter"]

    self.src_ip = yml_input["ip_address"]["src"]
    self.dst_ip = yml_input["ip_address"]["dst"]

    self.use_timediff = yml_input["timestamp"]["use_timediff"]
    self.timestamp_idx = yml_input["timestamp"]["idx"]
    print(self.timestamp_idx, type(self.timestamp_idx))
    self.timestamp_format = yml_input["timestamp"]["format"]

    for f_idx, f_type in yml_input["used_features"].items():
      self.feature_map[f_idx] = self.__feature_type_mapping[f_type]

    self.label_idx = yml_input["labeling"]["idx"]
    self.background_label = yml_input["labeling"]["background_label"]

  def __str__(self) -> str:
    return "TODO: Print out the loaded settings for better usability of program"