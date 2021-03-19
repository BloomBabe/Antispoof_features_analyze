# Based on this paper http://rose1.ntu.edu.sg/datasets/RecaptureImages/LCD_Recaptured_Cao2010.pdf

import numpy as np
import cv2
import pandas as pd
from skimage import color
from skimage.feature import local_binary_pattern


def lbp_features(img_bgr, label, filename,  df=None):
    """
    :param image: input 3-dim image
    :param df: features np.array, if None create a new np.array
    :return: np.array with multi-scale lbp histograms
    """
    # list of tuples PR params of Local binary pattern operator,
    # where P is the dimension of the angular space and R determines the resolution
    scales = [(8, 1), (24, 3)]
    n_bins = [10, 26]

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0.5)
    COLOR_MODE = ['BGR', 'HSV', 'YCR', 'GRAY']
    # compute coefficients
    for num_mode, img in enumerate([img_bgr, img_hsv, img_ycrcb, img_gray]):
        for num_channel, channel in enumerate(cv2.split(img)):
            for i, scale in enumerate(scales):
                for num_bins in n_bins:
                    patterns = local_binary_pattern(channel, scale[0], scale[1], method='uniform')
                    hist, _ = np.histogram(patterns, bins=num_bins, range=(0, num_bins), density=True)

                    labels = []
                    for num in range(len(hist)):
                        column_name = f'LBP_{COLOR_MODE[num_mode]}_CH{num_channel}_SC{scale[0]}+{scale[1]}_COUNTBINS{num_bins}_NUMBIN{num}'
                        labels.append(column_name)

                    feat = hist.reshape((1, -1))
                    feat_vector = pd.DataFrame(feat, columns=labels)
                    if df is None:
                        df = feat_vector.copy()
                        id_df = pd.DataFrame([[label, filename]], columns=['label', 'filename'])
                        df = pd.concat([id_df, df], axis=1)
                    else:
                        df = pd.concat([df, feat_vector], axis=1)
    return df
