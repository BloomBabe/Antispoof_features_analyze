import numpy as np
import argparse
import cv2
import os
import pandas as pd

from tqdm import tqdm
from features_extract.lbp_feat  import lbp_features
from features_extract.msws import wavelet_features


data_dir = 'D:\\JetBrains_PyCharm\\ml_university\\ml_antispoof\\data'
feat_table = 'D:\\JetBrains_PyCharm\\ml_university\\ml_antispoof'


data_frame = None

for label in os.listdir(data_dir):
    if label == 'real':
        binary_label = 1
    elif label == 'spoof':
        binary_label = 0
    label_dir = os.path.join(data_dir, label)

    for file_name in tqdm(os.listdir(label_dir)):
        img_pth = os.path.join(label_dir, file_name)
        try:
            img = cv2.imread(img_pth)
        except:
            print('Image did not load', file_name)
            continue

        feat_obj = wavelet_features(img, label=binary_label, filename=file_name)
        feat_obj = lbp_features(img, label=binary_label, filename=file_name, df=feat_obj)
        if data_frame is None:
            data_frame = feat_obj.copy()
        else:
            data_frame = pd.concat([data_frame, feat_obj], axis=0)

if data_frame is not None:
    data_frame.to_csv(os.path.join(feat_table, 'my_features.csv'), index=False)
else:
    print('DataFrame is empty.')


