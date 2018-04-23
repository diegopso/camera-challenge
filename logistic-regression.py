from sklearn.linear_model import LogisticRegression
import pandas as pd

if __name__ == '__main__':
	df = pd.read_csv('data/train-wavelet-features.csv')
	regressor = LogisticRegression().fit()
