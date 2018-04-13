import re, os, sys, random
import multiprocessing as mp
import numpy as np
import pandas as pd
import scipy.misc, scipy.stats
from sklearn.decomposition import PCA

def extract_features(filename, batches=3):
    label = re.search(r'(.*)\/\((.*?)\)[\d]+\.[a-zA-Z]{3}', filename).group(2)
    image = scipy.misc.imread(filename)
    
    w = int(image.shape[0] / 512)
    h = int(image.shape[1] / 512)

    metadata = []
    for b in range(0, batches):
        i = random.randint(0, w-1)
        j = random.randint(0, h-1)

        sample_image = image[i*512:(i+1)*512,j*512:(j+1)*512,:]
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
        pca = PCA(n_components=4)
        pca.fit(np.asarray(channels))
        metadata.append(pca.explained_variance_ratio_)
    df = pd.DataFrame(metadata)
    df.columns = ['ccpca_1','ccpca_2','ccpca_3','ccpca_4']
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

        pool = mp.Pool(pool_size)
        metadata = pool.map(extract_features, [os.path.join(folder, f) for f in os.listdir(folder)])
        results.extend(metadata)

    df = pd.concat(results, ignore_index=True)
    df.to_csv('data/train-crosscorrelation-features.csv', index=False)