"""
The main file of the real time system. Receives all the input parameters and start the system.
"""

import os
import argparse

from model_training import ModelFactory, ModelTrainer
from data_handling import DataMapper
from evaluation import ModelEvaluator

def check_paths(args):
  """
  Checks the path parameters provided when running the program.

  args: Arguments as returned by argparse parser.
  """

  inf_path = args.infile
  if not os.path.isfile(inf_path):
    raise ValueError("File does not exist: {}".format(inf_path))

  outf_path = args.outf_path
  if not os.path.isdir(outf_path):
    print("Creating directory for output: {}".format(outf_path))
    os.mkdir(outf_path)

  settings_path = args.settingsfile
  if not os.path.isfile(settings_path):
    print("Settings file does not exist: {}".format(settings_path))
    os.mkdir(settings_path)

  return inf_path, outf_path, settings_path


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # some paths
  parser.add_argument("infile", type=str, help="Path to input file")
  parser.add_argument("settingsfile", type=str, help="Path to settings file for input data. See ./data_settings/")
  parser.add_argument("--outf_path", type=str, default="results", help="Where to write output")

  # Model and model training related arguments
  parser.add_argument("--model", type=str, default="vanilla_ae", help="The type of model to use. \
                      Currently only 'vanilla_ae' is supported.")
  parser.add_argument("--n_training_examples", type=int, default=int(1e6), help="Number of training examples")
  parser.add_argument("--b_size", type=int, default=1, help="Batch-size for training. If n_training_examples mod b_size not zero the\
                      last batch will just be filled with the remaining training examples.")
  parser.add_argument("--use_cuda", type=bool, default=False, help="1 if cuda is used, 0 otherwise")
  parser.add_argument("--benign_training", type=bool, default=True, help="If True (1), then only benign examples are used during training.\
                      Else, all data examples will be used during training.")

  # Data related arguments
  parser.add_argument("--ngram_size", type=int, default=5, help="The size of the ngrams used.")

  # evaluation related arguments
  parser.add_argument("--save_model", type=bool, default=True, help="Saves the trained model.")
  parser.add_argument("--model_save_path", type=str, default=os.path.join("evaluation", "trained_models", "trained_model.keras"), help="The\
                      full path where to save the trained model.")

  args = parser.parse_args()

  # preprocessing
  inf_path, outf_path, settings_path = check_paths(args)
  data_handler = DataMapper(inf_path, settings_path, args.ngram_size)

  # prepare training
  if not args.use_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU

  model_type = args.model
  input_shape = data_handler.get_input_shape()
  model_factory = ModelFactory(model_type)
  model = model_factory.get_model(input_shape=input_shape)

  # train model
  print("Expected input shape for model: {}".format(input_shape))
  trainer = ModelTrainer(data_handler, model_type, args.n_training_examples, args.save_model, args.model_save_path)
  trainer.train(model, args.benign_training, b_size=args.b_size)

  # model evaluation
  evaluator = ModelEvaluator(model_type, data_handler)
  evaluator.evaluate(model)