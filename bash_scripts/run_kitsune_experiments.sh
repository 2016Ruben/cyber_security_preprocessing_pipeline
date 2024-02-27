
# input parameters
ngram_size=5
yaml_input="data_handling/settings/kitsune_settings.yml"
# b_size=8 along with shuffle=True to get the best results so far 
b_size=1
model_type="vanilla_ae" # "vanilla

cd ..
for dir in data/kitsune_data/*/; do
  dsname=$(basename "$dir")
  echo "Starting with dataset ${dsname}\n"
  mkdir -p "results/${dsname}" # time cannot create directories

  pcapf="${dir}${dsname}_pcap.pcapng"
  labelf="${dir}${dsname}_labels.csv"

  for it in {1..10}; do
    figure_file="results/${dsname}/auc_${it}.png"
    model_file="results/${dsname}/model_${it}.keras"
    res_file="results/${dsname}/results_dict_${it}.pk"
    time_file="results/${dsname}/time_${it}.txt"

    { time python main.py ${pcapf} ${yaml_input} --input_type=kitsune --model_type=${model_name} --labelf_path=${labelf} --ngram_size=${ngram_size} --save_scaler=0 --figure_path=${figure_file} --model_save_path=${model_file} --results_dict_save_path=${res_file} --b_size=${b_size} --n_training_examples=1000000 ; } 2> ${time_file}
  done
done