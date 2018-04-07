import scipy.misc, scipy.stats
import pywt 
import math 
import numpy as np
import pandas as pd
import sys, os, re, random
import multiprocessing as mp

def savefig(data, location):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,12))
    plt.imshow(data)
    plt.savefig(location)

def image_statistics(data):
    flat = data.flatten()
    mean = np.mean(flat)
    variance = np.mean(abs(flat - mean)**2)
    skewness = scipy.stats.skew(flat)
    kurtosis = scipy.stats.kurtosis(flat)
    return [mean, variance, skewness, kurtosis]

def extract_features(filename, batches=3):
    image = scipy.misc.imread(filename, flatten=True)
    label = re.search(r'(.*)\/\((.*?)\)[\d]+\.jpg', filename).group(2)

    w = int(image.shape[0] / 512)
    h = int(image.shape[1] / 512)

    metadata = []
    for k in range(0, batches):
        i = random.randint(0, w-1)
        j = random.randint(0, h-1)

        sample_image = image[i*512:(i+1)*512,j*512:(j+1)*512]

        wavelet = pywt.Wavelet('haar')
        (cA, (cH, cV, cD)) = pywt.dwt2(sample_image, wavelet)

        m = [label]
        m.extend(image_statistics(sample_image))
        m.extend(image_statistics(cH))
        m.extend(image_statistics(cV))
        m.extend(image_statistics(cD))

        metadata.append(m)

    df = pd.DataFrame(metadata)
    df.columns = [
        'label', 
        'original_mean', 'original_variance', 'original_skewness', 'original_kurtosis',
        'wavelet_h_mean', 'wavelet_h_variance', 'wavelet_h_skewness', 'wavelet_h_kurtosis',
        'wavelet_v_mean', 'wavelet_v_variance', 'wavelet_v_skewness', 'wavelet_v_kurtosis',
        'wavelet_d_mean', 'wavelet_d_variance', 'wavelet_d_skewness', 'wavelet_d_kurtosis',
        ]
    
    return df

if __name__ == '__main__':
    train_dir = 'data/train/'
    pool_size = 3
    results = []
    for d in os.listdir(train_dir):
        folder = os.path.join(train_dir, d)
        pool = mp.Pool(pool_size)
        metadata = pool.map(extract_features, [os.path.join(folder, f) for f in os.listdir(folder)])
        results.extend(metadata)

    df = pd.concat(results, ignore_index=True)
    df.to_csv('data/train-wavelet-features.csv', index=False)