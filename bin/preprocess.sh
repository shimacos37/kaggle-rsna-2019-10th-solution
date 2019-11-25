mkdir -p intermediate_output/preprocessed_data
# appian's preprocess
cd models/base_cnn/model_base
sh bin/preprocess.sh
cd ../../../

# ricky's preprocess
cd models/base_cnn/ricky_senet154_customlabels
sh bin/preprocess.sh
cd ../../../

# shimacos's preprocess
cd models/base_cnn/shimacos_models
sh bin/preprocess.sh
cd ../../../
