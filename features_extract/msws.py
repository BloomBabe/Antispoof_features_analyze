# Based on this paper http://rose1.ntu.edu.sg/datasets/RecaptureImages/LCD_Recaptured_Cao2010.pdf
import pandas as pd
import cv2
import numpy as np
import pywt
from sklearn.preprocessing import normalize


def wavelet_features(img_bgr, label, filename, color_mode='RGB',  df=None):
    """
    :param image: input 3-dim image
    :param level: level of wavelet decomposition
    :param df: features np.array, if None create a new np.array
    :return: np.array with Multi-Scale Wavelet Statistics
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
    COLOR_MODE = ['BGR', 'HSV', 'YCR']
    # compute coefficients
    for num_mode, img in enumerate([img_bgr, img_hsv, img_ycrcb]):
        for num_ch, channel in enumerate(cv2.split(img)):
            coeffs = pywt.wavedec2(channel, 'haar', mode='periodization', level=1)
            # normalize each coefficient array independently for better visibility
            decomp = ['LH', 'HL', 'HH']
            for i, c in enumerate(coeffs[1][:]):
                feat = np.array([np.std(np.abs(c)), np.mean(np.abs(c))]).reshape((1, -1))
                labels = [f'WS_{COLOR_MODE[num_mode]}_CH{num_ch}_{decomp[i]}_std',
                          f'WS_{COLOR_MODE[num_mode]}_CH{num_ch}_{decomp[i]}_mean']
                feat_vector = pd.DataFrame(feat, columns=labels)
                if df is None:
                    df = feat_vector.copy()
                    id_df = pd.DataFrame([[label, filename]], columns=['label', 'filename'])
                    df = pd.concat([id_df, df], axis=1)
                else:
                    df = pd.concat([df, feat_vector], axis=1)
    # df = normalize(df)
    return df
