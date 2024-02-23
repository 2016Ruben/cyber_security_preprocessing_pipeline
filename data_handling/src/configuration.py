"""
Written by Robert Baumgartner, 2024
r.baumgartner-1@tudelft.nl

For details, see ../readme.md
"""

import os
import yaml

"""
Used by all configs, hence made available throughout file.
TODO: Is it a good choice to make this one a "global" dict?
"""
_feature_type_mapping = {
  "continuous": 0,
  "categorical": 1
}

class FileInputConfigs():
  """
  Configurations for when reading from an input file of structured data,
  e.g. Netflow formatted data.
  """
  def __init__(self):
    self.has_header = None
    self.delimiter = None
    
    self.src_ip = None
    self.dst_ip = None

    self.timestamp_idx = None
    self.timestamp_format = None

    self.feature_map = dict() # mapping index to type. 0 for continuous, 1 for categorical

    self.label_idx = None
    self.background_label = None
    self.labelf_has_header = None
    self.labelf_delimiter = None

  def read_settings(self, infile: str):
    if not os.path.isfile(infile):
      raise ValueError("Settings file does not exist: {}".format(infile))

    # for details on Loader in next line: https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
    yml_input = yaml.load(open(infile, "rt"), Loader=yaml.SafeLoader) 
    
    self.has_header = yml_input["structure"]["has_header"]
    self.delimiter = yml_input["structure"]["delimiter"]

    self.src_ip = yml_input["ip_address"]["src"]
    self.dst_ip = yml_input["ip_address"]["dst"]

    self.timestamp_idx = yml_input["timestamp"]["idx"]
    self.timestamp_format = yml_input["timestamp"]["format"]

    for f_idx, f_type in yml_input["used_features"].items():
      self.feature_map[f_idx] = _feature_type_mapping[f_type]

    self.label_idx = yml_input["labeling"]["idx"]
    self.background_label = yml_input["labeling"]["background_label"]
    self.labelf_has_header = yml_input["labeling"]["filestructure"]["has_header"]
    self.labelf_delimiter = yml_input["labeling"]["filestructure"]["delimiter"]


  def __str__(self) -> str:
    return "TODO: Print out the loaded settings for better usability of program"
  

class StreamInputConfigs():
  """
  Configurations for when reading from a data stream.
  """
  def __init__(self):
    self.src_ip = None
    self.dst_ip = None

    self.timestamp_idx = None
    self.timestamp_format = None

    self.feature_map = dict() # mapping index to type. 0 for continuous, 1 for categorical

    self.label_idx = None
    self.background_label = None
    self.labelf_has_header = None
    self.labelf_delimiter = None
    
  def read_settings(self, infile: str):
    if not os.path.isfile(infile):
      raise ValueError("Settings file does not exist: {}".format(infile))

    # for details on Loader in next line: https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
    yml_input = yaml.load(open(infile, "rt"), Loader=yaml.SafeLoader) 
    
    self.src_ip = yml_input["ip_address"]["src"]
    self.dst_ip = yml_input["ip_address"]["dst"]

    self.timestamp_idx = yml_input["timestamp"]["idx"]
    self.timestamp_format = yml_input["timestamp"]["format"]

    for f_idx, f_type in yml_input["used_features"].items():
      self.feature_map[f_idx] = _feature_type_mapping[f_type]

    self.label_idx = yml_input["labeling"]["idx"]
    self.background_label = yml_input["labeling"]["background_label"]
    self.labelf_has_header = yml_input["labeling"]["filestructure"]["has_header"]
    self.labelf_delimiter = yml_input["labeling"]["filestructure"]["delimiter"]

  def __str__(self) -> str:
    return "TODO: Print out the loaded settings for better usability of program"