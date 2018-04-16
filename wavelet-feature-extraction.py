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
    label = re.search(r'(.*)\/\((.*?)\)[\d]+\.[a-zA-Z]{3}', filename).group(2)
    image = scipy.misc.imread(filename)

    w = image.shape[0]
    h = image.shape[1]

    pad1 = int((w - 512) / 2)
    pad2 = int((h - 512) / 2)

    batches = [
        [(0,0),(512,512)], #1
        [(w-512,0),(w, 512)], #2
        [(0,h-512),(512, h)], #3
        [(w-512,h-512),(w, h)], #4
        [(pad1, pad2),(pad1+512, pad2+512)] #5
        ]

    metadata = []
    batch_index = 0
    for b in batches:
        batch_index = batch_index + 1
        for channel in range(0, 3):
            sample_image = image[b[0][0]:b[1][0],b[0][1]:b[1][1],channel]

            wavelet = pywt.Wavelet('db1')
            # (cA, cD) = pywt.dwt(sample_image, wavelet)
            (cA, (cH, cV, cD)) = pywt.dwt2(sample_image, wavelet)

            # scipy.misc.imsave('data/sample_image.png', sample_image)
            # scipy.misc.imsave('data/cA.png', cH)

            # cH = sample_image - cH
            # cV = sample_image - cV
            # cD = sample_image - cD

            m = [filename,label,channel,batch_index]
            # m.extend(image_statistics(sample_image))
            m.extend(image_statistics(cH))
            m.extend(image_statistics(cV))
            m.extend(image_statistics(cD))

            metadata.append(m)

    df = pd.DataFrame(metadata)
    df.columns = [
        'filename', 'label', 'channel', 'batch',
        # 'original_mean', 'original_variance', 'original_skewness', 'original_kurtosis',
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
        
        #single test
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