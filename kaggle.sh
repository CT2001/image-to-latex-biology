#!/bin/bash

cp /opt/conda/lib/libstdc++.so.6.0.29 /usr/lib/x86_64-linux-gnu/
cd /usr/lib/x86_64-linux-gnu/
rm -rf libstdc++.so.6
ln -s libstdc++.so.6.0.29 libstdc++.so.6
cd /kaggle/working
pip install hydra-core --upgrade
pip install editdistance
mkdir -p image-to-latex/data/formula_labels_oneline
mkdir -p image-to-latex/data/formula_labels_noerror
mkdir -p image-to-latex/data/formula_labels
mkdir -p image-to-latex/data/formula_images
cd image-to-latex/data-preprocess/
python data_filter.py
rm -r -f '../data/formula_labels_oneline'
python no_chinese.py
rm -r -f '../data/formula_labels_noerror'
python extract_image_according_to_label_list.py
python data_preprocess_for_im2latex.py
cd ..
chmod +x scripts/find_and_replace.sh
scripts/find_and_replace.sh data/im2latex_formulas.norm.lst data/im2latex_formulas.norm.new.lst
python scripts/prepare_data.py --project_path='/kaggle/working/image-to-latex'
