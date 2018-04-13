import scipy.misc, scipy.stats
import pywt 
import math 
import numpy as np
import pandas as pd
import sys, os, re, random
import multiprocessing as mp
from sklearn.decomposition import PCA

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

def crosscorrelation_extraction(image):
    channels = []
    for i in range(0, 3):
        channel1 = image[:,:,i]
        for j in range(0, 3):
            channel2 = image[:,:,j]
            features = []
            for d1 in range(0, 3):
                for d2 in range(0, 3):
                    if d1 == 0 and d2 == 0:
                        continue

                    r1 = []
                    r2 = []
                    for k in range(0, 512):
                        for l in range(0, 512):
                            kd = k + d1
                            ld = l + d2
                            r1.append(channel1[k,l])
                            r2.append(channel2[kd,ld] if kd < 512 and ld < 512 else 0)
                    features.append(scipy.stats.pearsonr(r1, r2)[0])
            channels.append(features)
    pca = PCA(n_components=4)
    pca.fit(np.asarray(channels))
    return pca.explained_variance_ratio_

def wavelet_extraction(image):
    wavelet = pywt.Wavelet('haar')
    (cA, (cH, cV, cD)) = pywt.dwt2(image, wavelet)

    m = []
    m.extend(image_statistics(image))
    m.extend(image_statistics(cH))
    m.extend(image_statistics(cV))
    m.extend(image_statistics(cD))

    return m

def extract_features(filename, batches=3):
    label = re.search(r'(.*)\/\((.*?)\)[\d]+\.[a-zA-Z]{3}', filename).group(2)
    image = scipy.misc.imread(filename, flatten=True)
    image3d = scipy.misc.imread(filename)

    w = int(image.shape[0] / 512)
    h = int(image.shape[1] / 512)

    metadata = []
    for k in range(0, batches):
        i = random.randint(0, w-1)
        j = random.randint(0, h-1)
        sample_image = image[i*512:(i+1)*512,j*512:(j+1)*512]
        sample_image3d = image3d[i*512:(i+1)*512,j*512:(j+1)*512,:]

        features = wavelet_extraction(sample_image)
        features.extend(crosscorrelation_extraction(sample_image3d))
        metadata.append(features)

    df = pd.DataFrame(metadata)
    df.columns = [
        'original_mean', 'original_variance', 'original_skewness', 'original_kurtosis',
        'wavelet_h_mean', 'wavelet_h_variance', 'wavelet_h_skewness', 'wavelet_h_kurtosis',
        'wavelet_v_mean', 'wavelet_v_variance', 'wavelet_v_skewness', 'wavelet_v_kurtosis',
        'wavelet_d_mean', 'wavelet_d_variance', 'wavelet_d_skewness', 'wavelet_d_kurtosis',
        'ccpca_1','ccpca_2','ccpca_3','ccpca_4',
        ]

    df['label'] = label
    return df

if __name__ == '__main__':
    train_dir = 'data/train/'
    pool_size = 3
    results = []
    for d in os.listdir(train_dir):
        folder = os.path.join(train_dir, d)

        # single test
        # frame = extract_features(os.path.join(folder, os.listdir(folder)[0]))

        # paralel test
        pool = mp.Pool(pool_size)
        metadata = pool.map(extract_features, [os.path.join(folder, f) for f in os.listdir(folder)])
        results.extend(metadata)

    df = pd.concat(results, ignore_index=True)
    df.to_csv('data/train-features.csv', index=False)