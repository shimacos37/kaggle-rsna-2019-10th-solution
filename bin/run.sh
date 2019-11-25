#!/bin/bash

# base CNN models training and predicting
cd models/base_cnn/model_base
sh bin/train.sh
sh bin/predict.sh
cd ../../../

# ricky's model
cd models/base_cnn/ricky_se_resnext_410
sh bin/train.sh
sh bin/predict.sh
cd ../../../

cd models/base_cnn/ricky_se_resnext101_mixup
sh bin/train.sh
sh bin/predict.sh
cd ../../../

cd models/base_cnn/ricky_senet154_customlabels
sh bin/train.sh
sh bin/predict.sh
cd ../../../

# shimacos's model
cd models/base_cnn/shimacos_models
sh bin/train_001.sh
sh bin/test_001.sh
sh bin/train_002.sh
sh bin/test_002.sh
sh bin/train_003.sh
sh bin/test_003.sh
cd ../../../

# sugawarya's model
cd models/base_cnn/sugawara_efficientnetb3
sh bin/train.sh
sh bin/predict.sh
cd ../../../

# 2kyym's model
cd models/base_cnn/2kyym_inceptionv4
sh bin/train.sh
sh bin/predict.sh
cd ../../../

cd models/base_cnn/2kyym_inception_resnet_v2
sh bin/train.sh
sh bin/predict.sh
cd ../../../

cd models/base_cnn/2kyym_xception
sh bin/train.sh
sh bin/predict.sh
cd ../../../

# 1st stacking training
cd models/first_stacking
sh ./bin/stacking.sh
cd ../../

# 2nd stacking training
cd models/second_stacking
sh ./bin/stacking.sh

# make submission
sh ./bin/submission.sh
cd ../../