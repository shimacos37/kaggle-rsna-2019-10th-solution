#!/bin/bash

python cnn_stacking_1.py
python cnn_stacking_2.py
python preprocess_for_sugawara_stacking1.py
python get_feats_for_first_stacking_sugawara.py
python lgbm_first_stacking.py
python mlp_stacking.py
