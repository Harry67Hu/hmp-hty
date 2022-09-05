import numpy as np
import copy
from sklearn import datasets

def pca(samples, target_dim):
    assert len(samples.shape) == 2
    data = samples - np.mean(samples,axis=0)  # mean at batch dim
    covMat = np.cov(data,rowvar=0)
    fValue,fVector = np.linalg.eig(covMat)
    fValueSort = np.argsort(-fValue)
    fValueTopN = fValueSort[:target_dim]
    fvectormat = fVector[:,fValueTopN]
    down_dim_data = np.dot(data, fvectormat)
    return down_dim_data

data = datasets.load_iris()["data"] # (batch=150, 4)
data2 = np.random.rand(150, 500)
data = np.concatenate((data, data2), axis=-1)
data = data[:10, :]

res = pca(samples=data, target_dim=3)
print(data.shape, res.shape)