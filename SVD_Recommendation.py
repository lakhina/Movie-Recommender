"""
    Author: Chetna, Pallavi , Ruchika
    Inst: IIIT- Delhi, 2017-2019
    Task: Recommendation on Movie Lens Dataset (Reference evaluation in svd.py)
    Ref: Code help taken from "Google" / "github" / "Stack overflow" / "analytics_vidhya" / "Kaggle"  for SVD
"""

# --------------imports-----------
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import math
import copy

#----------------variables-----------
users=943
movies=1682
k=6

#------------------read data--------------
df = pd.read_csv('ml-100k/ua.base', sep="\t", header=-1)
vals = df.values
vals[:, 0:2] -= 1
train = scipy.sparse.csr_matrix((vals[:, 2], (vals[:, 0], vals[:, 1])), dtype=np.float, shape=(users,movies))

df = pd.read_csv('ml-100k/ua.base', sep="\t", header=-1)
values = df.values
values[:, 0:2] -= 1
test = scipy.sparse.csr_matrix((vals[:, 2], (vals[:, 0], vals[:, 1])), dtype=np.float, shape=(users,movies))

train_rows, train_cols = train.nonzero()
test_rows, test_cols = test.nonzero()

#--------------normalization--------------------
trainMean = np.multiply(np.sum(train,axis=1),1/np.sum(train!=0,axis=1))
trainMean = np.array([(i+1 if i==0 else i) for i in list(trainMean)])

normalizedTrain=copy.deepcopy(train)

for i in range(0,len(train_rows)):
    normalizedTrain[train_rows[i], train_cols[i]] -= trainMean[train_rows[i]]

testMean = np.multiply(np.sum(test,axis=1),1/np.sum(test!=0,axis=1))
testMean = np.array([(i+1 if i==0 else i) for i in list(testMean)])

normalizedTest=copy.deepcopy(test)

for i in range(0,len(test_rows)):
    normalizedTest[test_rows[i], test_cols[i]] -= testMean[test_rows[i]]

#---------------SVD transformation for train-----------------
U, sigma, Vt = svds(normalizedTrain)

pred = U[:, 0:k].dot(np.diag(sigma[0:k])).dot(Vt[0:k, :])

# ------------save Model--------
np.savetxt('modelSVD1', pred, delimiter=',',fmt='%f')
np.savetxt('modelSVD2',train_rows,delimiter=',',fmt='%f')
np.savetxt('modelSVD3',train_cols,delimiter=',',fmt='%f')

#-----------Recommend------------
def last(n):
    return n[-1]

def recommendSVD(userID,count):
    user=userID-1
    AlreadymovieList=[]
    pred=np.loadtxt('modelSVD1', delimiter=',')
    train_rows=np.loadtxt('modelSVD2', delimiter=',')
    train_cols=np.loadtxt('modelSVD3', delimiter=',')
    userRatings=pred[user]
    for idx,val in enumerate(train_rows):
        if val==user:
            AlreadymovieList.append(train_cols[idx])
    Movie_Ratgs=[]
    for idx,val in enumerate(userRatings):
        if not(idx in AlreadymovieList):
            Movie_Ratgs.append((idx+1,val))
    
    return sorted(Movie_Ratgs, key=last,reverse=True)[:min(count,len(Movie_Ratgs))]
    
    
##predictions = recommendSVD(1310,5)

