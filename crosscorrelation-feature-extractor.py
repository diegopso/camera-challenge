import re, os, sys, random
import multiprocessing as mp
import numpy as np
import pandas as pd
import scipy.misc, scipy.stats
from sklearn.decomposition import PCA

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
        sample_image = image[b[0][0]:b[1][0],b[0][1]:b[1][1],:]
        batch_index = batch_index + 1
        channels = []
        for i in range(0, 3):
            channel1 = sample_image[:,:,i]
            for j in range(0, 3):
                channel2 = sample_image[:,:,j]
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
        
        channels = np.asarray(channels)

        features = np.asarray([0,0,0,0])
        if not np.any(np.isnan(channels)) and not np.any(np.isinf(channels)):
            pca = PCA(n_components=4)
            pca.fit(np.asarray(channels))
            features = pca.explained_variance_ratio_

        m = [batch_index]
        m.extend(features)
        metadata.append(m)

    df = pd.DataFrame(metadata)
    df.columns = ['batch', 'ccpca_1','ccpca_2','ccpca_3','ccpca_4']
    df['label'] = label
    df['filename'] = filename
    return df

def train(pool_size=3):
    train_dir = 'data/train/'
    results = []
    for d in os.listdir(train_dir):
        folder = os.path.join(train_dir, d)
        
        # single test
        # frame = extract_features(os.path.join(folder, os.listdir(folder)[0]))
        # print(frame)
        # sys.exit()

        # paralel test
        pool = mp.Pool(pool_size)
        metadata = pool.map(extract_features, [os.path.join(folder, f) for f in os.listdir(folder)])
        results.extend(metadata)

    df = pd.concat(results, ignore_index=True)
    df.to_csv('data/train-crosscorrelation-features.csv', index=False)

def test(pool_size=3):
    folder = 'data/test/'

    # single test
    # frame = extract_features(os.path.join(folder, os.listdir(folder)[0]))
    # print(frame)
    # sys.exit()

    # paralel test
    pool = mp.Pool(pool_size)
    metadata = pool.map(extract_features, [os.path.join(folder, f) for f in os.listdir(folder)])
    results.extend(metadata)

    df = pd.concat(results, ignore_index=True)
    df.to_csv('data/test-crosscorrelation-features.csv', index=False)

if __name__ == '__main__':
    test(3)