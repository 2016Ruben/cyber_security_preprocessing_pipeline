
# input parameters
ngram_size=5
yaml_input="data_handling/settings/kitsune_settings.yml"
# b_size=8 along with shuffle=True to get the best results so far 
b_size=1
model_type="vanilla_ae" # "vanilla
output_dir="vanilla_ae"

cd ..
for dir in data/kitsune_data/*/; do
  dsname=$(basename "$dir")
  echo "Starting with dataset ${dsname}\n"
  mkdir -p "${output_dir}/${dsname}" # time cannot create directories

  pcapf="${dir}${dsname}_pcap.pcapng"
  labelf="${dir}${dsname}_labels.csv"

  for it in {1..1}; do
    figure_file="${output_dir}/${dsname}/auc_${it}.png"
    model_file="${output_dir}/${dsname}/model_${it}.keras"
    res_file="${output_dir}/${dsname}/results_dict_${it}.pk"
    time_file="${output_dir}/${dsname}/time_${it}.txt"

    { time python main.py ${pcapf} ${yaml_input} --input_type=kitsune --model_type=${model_type} --labelf_path=${labelf} --ngram_size=${ngram_size} --save_scaler=0 --figure_path=${figure_file} --model_save_path=${model_file} --results_dict_save_path=${res_file} --b_size=${b_size} --n_training_examples=1000000 ; } 2> ${time_file}
  done
done