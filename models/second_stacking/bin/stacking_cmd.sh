#!/bin/bash

python preprocess_for_sugawara_stacking2.py
python get_feats_for_2nd_stacking.py
python lgbm_second_stacking.py
