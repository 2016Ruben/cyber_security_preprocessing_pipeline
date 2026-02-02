"""
The main file of the real time system. Receives all the input parameters and starts the system.
"""

import os
import argparse

from model_training import ModelFactory, ModelTrainer, ScalerWrapper
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
    raise ValueError("Settings file does not exist: {}".format(settings_path))

  trained_model = args.trained_model
  if trained_model is not None and not os.path.isfile(trained_model):
      raise ValueError("Could not find trained model at location {}".format(trained_model))

  figure_path = args.figure_path
  if not os.path.isdir(figure_path):
    print("Creating directory for figures: {}".format(figure_path))
    os.mkdir(figure_path)

  return inf_path, outf_path, settings_path, figure_path


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # some paths
  parser.add_argument("infile", type=str, help="Path to input file")
  parser.add_argument("settingsfile", type=str, help="Path to settings file for input data. See ./data_settings/")
  parser.add_argument("--outf_path", type=str, default="results", help="Where to write output")

  # Model and model training related arguments
  parser.add_argument("--model", type=str, default="vanilla_ae", help="The type of model to use. \
                      Currently only 'vanilla_ae' is supported.")
  parser.add_argument("--trained_model", type=str, default=None, help="If provided, then this will be a full path to a trained model. Skipping\
                      training and going right into model evaluation.")
  parser.add_argument("--n_training_examples", type=int, default=int(1e6), help="Number of training examples")
  parser.add_argument("--b_size", type=int, default=1, help="Batch-size for training. If n_training_examples mod b_size not zero the\
                      last batch will just be filled with the remaining training examples.")
  parser.add_argument("--use_cuda", type=bool, default=False, help="1 if cuda is used, 0 otherwise")
  parser.add_argument("--benign_training", type=bool, default=True, help="If True (1), then only benign examples are used during training.\
                      Else, all data examples will be used during training.")
  parser.add_argument("--scale_data", type=bool, default=True, help="If True (1) then MinMax-Scaling will be applied.")

  # Data related arguments
  parser.add_argument("--ngram_size", type=int, default=5, help="The size of the ngrams used.")
  parser.add_argument("--channels", type=str, default="all", help="Select channels: all, seq, src, dst, conn")

  # evaluation related arguments
  parser.add_argument("--max_eval_samples", type=int, default=None, help="The maximum number of samples that we want to evaluate on.\
                      If None do an exhaustive evaluation through the input file.")
  parser.add_argument("--evaluation_bsize", type=int, default=int(1e5), help="The batch size during evaluation. Trades off speed of evaluation with\
                      memory consumption. Batches of input data are gathered before given the model to evaluate.")

  # saving and loading
  parser.add_argument("--save_model", type=bool, default=True, help="If true it saves the trained model.")
  parser.add_argument("--save_scaler", type=bool, default=True, help="If true it saves the MinMaxScaler. Only applied when scale_data\
                      is true.")
  parser.add_argument("--save_results", type=bool, default=True, help="Saves the evaluator model.")
  parser.add_argument("--model_save_path", type=str, default=os.path.join("results", "trained_models", "trained_model.keras"), help="The\
                      full path where to save the trained model.")
  parser.add_argument("--scaler_save_path", type=str, default=os.path.join("results", "trained_models", "minmaxscaler.pk"), help="The\
                      full path where to save the minmax-scaler.")
  parser.add_argument("--results_dict_save_path", type=str, default=os.path.join("results", "results_dict.pk"), help="The\
                      full path where to save the results as a dictionary.")
  parser.add_argument("--figure_path", type=str, default=os.path.join("results", "figures"), help="The path where we store the figures in.")  
  
  args = parser.parse_args()

  # preprocessing
  inf_path, outf_path, settings_path, figure_path = check_paths(args)
  data_handler = DataMapper(inf_path, settings_path, args.ngram_size, channels=args.channels)

  # prepare training
  if not args.use_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU

  model_type = args.model
  input_shape = data_handler.get_input_shape()
  trained_model = args.trained_model
  model_factory = ModelFactory(model_type)
  model = model_factory.get_model(trained_model, input_shape=input_shape)

  scaler = ScalerWrapper() if args.scale_data else None

  if trained_model is None:
    # train model
    print("Expected input shape for model: {}".format(input_shape))
    trainer = ModelTrainer(data_handler, scaler, model_type, args.n_training_examples, args.save_model, args.model_save_path)
    trainer.train(model, args.benign_training, b_size=args.b_size)

  if trained_model is None and args.save_scaler: 
    # trained_model check because we do not accidentally overwrite scaler when we did not fit a new one 
    scaler.save_state(args.scaler_save_path)

  # model evaluation
  evaluator = ModelEvaluator(data_handler, scaler, model_type)
  evaluator.evaluate(model, args.max_eval_samples, args.evaluation_bsize)
  if args.save_results:
    evaluator.save_results(args.results_dict_save_path)
  evaluator.print_plots(figure_path)