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

def extract_features(filename):
    image = scipy.misc.imread(filename)
    label = re.search(r'(.*)\/\((.*?)\)[\d]+\.[a-zA-Z]{3}', filename)
    if label:
        label = label.group(2)
    else:
        label = 'unknown'
    
    w = image.shape[0]
    h = image.shape[1]

    if w == 512 and h == 512:
        batches = [[(0, 0), (512, 512)]]
    else:
        pad1 = int((w - 512) / 2)
        pad2 = int((h - 512) / 2)
        batches = [
            [(0, 0), (512, 512)], #1
            [(w-512, 0), (w, 512)], #2
            [(0, h-512), (512, h)], #3
            [(w-512, h-512), (w, h)], #4
            [(pad1, pad2), (pad1+512, pad2+512)] #5
            ]

    metadata = []
    batch_index = 0
    for b in batches:
        batch_index = batch_index + 1
        
        m = [filename,label,batch_index]
        for channel in range(0, 3):
            sample_image = image[b[0][0]:b[1][0],b[0][1]:b[1][1],channel]

            wavelet = pywt.Wavelet('db1')
            # (cA, cD) = pywt.dwt(sample_image, wavelet)
            # (cA, (cH, cV, cD)) = pywt.dwt2(sample_image, wavelet)
            # cA = pywt.wavedec(sample_image, wavelet)
            c = pywt.dwt2(sample_image, wavelet)
            (cA, (cH, cV, cD)) = c

            c[0][:] = 0
            noise = pywt.idwt2(c, wavelet)

            # scipy.misc.imsave('data/m.png', m)
            # sys.exit()

            m.extend(image_statistics(noise))
            m.extend(image_statistics(cH))
            m.extend(image_statistics(cV))
            m.extend(image_statistics(cD))

        metadata.append(m)

    df = pd.DataFrame(metadata)
    df.columns = [
        'filename', 'label', 'batch',
        'noise_mean_r', 'noise_variance_r', 'noise_skewness_r', 'noise_kurtosis_r',
        'wavelet_h_mean_r', 'wavelet_h_variance_r', 'wavelet_h_skewness_r', 'wavelet_h_kurtosis_r',
        'wavelet_v_mean_r', 'wavelet_v_variance_r', 'wavelet_v_skewness_r', 'wavelet_v_kurtosis_r',
        'wavelet_d_mean_r', 'wavelet_d_variance_r', 'wavelet_d_skewness_r', 'wavelet_d_kurtosis_r',
        'noise_mean_g', 'noise_variance_g', 'noise_skewness_g', 'noise_kurtosis_g',
        'wavelet_h_mean_g', 'wavelet_h_variance_g', 'wavelet_h_skewness_g', 'wavelet_h_kurtosis_g',
        'wavelet_v_mean_g', 'wavelet_v_variance_g', 'wavelet_v_skewness_g', 'wavelet_v_kurtosis_g',
        'wavelet_d_mean_g', 'wavelet_d_variance_g', 'wavelet_d_skewness_g', 'wavelet_d_kurtosis_g',
        'noise_mean_b', 'noise_variance_b', 'noise_skewness_b', 'noise_kurtosis_b',
        'wavelet_h_mean_b', 'wavelet_h_variance_b', 'wavelet_h_skewness_b', 'wavelet_h_kurtosis_b',
        'wavelet_v_mean_b', 'wavelet_v_variance_b', 'wavelet_v_skewness_b', 'wavelet_v_kurtosis_b',
        'wavelet_d_mean_b', 'wavelet_d_variance_b', 'wavelet_d_skewness_b', 'wavelet_d_kurtosis_b',
        ]
    
    return df

def train(pool_size=3):
    train_dir = 'data/train/'
    results = []
    for d in os.listdir(train_dir):
        folder = os.path.join(train_dir, d)
        
        # single test
        # f = os.listdir(folder)[0]
        # df = extract_features(os.path.join(folder, f))
        # print(df)
        # sys.exit()

        # paralel test
        pool = mp.Pool(pool_size)
        metadata = pool.map(extract_features, [os.path.join(folder, f) for f in os.listdir(folder)])
        results.extend(metadata)

    df = pd.concat(results, ignore_index=True)
    df.to_csv('data/train-wavelet-features.csv', index=False)

def test(pool_size=3):
    folder = 'data/test/'
    # single test
    # f = os.listdir(folder)[0]
    # df = extract_features(os.path.join(folder, f))
    # print(df)
    # sys.exit()

    # paralel test
    pool = mp.Pool(pool_size)
    metadata = pool.map(extract_features, [os.path.join(folder, f) for f in os.listdir(folder)])
    results.extend(metadata)

    df = pd.concat(results, ignore_index=True)
    df.to_csv('data/test-wavelet-features.csv', index=False)

if __name__ == '__main__':
    test(3)