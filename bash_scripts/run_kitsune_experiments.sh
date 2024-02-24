
yaml_input="data_handling/settings/kitsune_settings.yml"

cd ..
for dir in data/kitsune_data/*/; do
  dsname=$(basename "$dir")
  echo "Starting with dataset ${dsname}"

  pcapf="${dir}/${dsname}_pcap.pcapng"
  labelf="${dir}/${dsname}_labels.csv"
  outdir="results/${dsname}"

  python main.py ${pcapf} ${yaml_input} --input_type=kitsune --labelf_path=${labelf} --figure_path=${outdir} --b_size=8 --n_training_examples=1000000
done