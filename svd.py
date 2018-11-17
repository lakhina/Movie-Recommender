"""
    Author: Chetna, Pallavi , Ruchika
    Inst: IIIT- Delhi, 2017-2019
    Task: SVD on Movie Lens Dataset- Transformation/ Normalization/ Evaluation/ Finding K
    Ref: Code help taken from "Google" / "github" / "Stack overflow" / "analytics_vidhya" / "Kaggle"  for SVD
"""
#-------------Imports-------------------------
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import math
import copy
#---------------variables----------------------
users=943
movies=1682
lk,uk=2,50
#---------------read Data------------------------

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
ks=np.arange(lk,uk)

#-------------initializations----------------------------------
maeTrain=np.zeros(ks.shape[0])
maeTest=np.zeros(ks.shape[0])

rmseTrain=np.zeros(ks.shape[0])
rmseTest=np.zeros(ks.shape[0])

train_ratings=normalizedTrain[(train_rows,train_cols)]
test_ratings=normalizedTest[(test_rows,test_cols)]
    
print("For k=", "MAE- Test", "RMSE- Test")
for i, k in enumerate(ks):
    #------Pick along top k eigen vectors--------------...
    pred = U[:, 0:k].dot(np.diag(sigma[0:k])).dot(Vt[0:k, :])
    #-------------get train and test prediction----------
    pred_train_ratings = pred[(train_rows,train_cols)]
    pred_test_ratings = pred[(test_rows,test_cols)]
    
    pred_train_ratings=np.reshape(pred_train_ratings.T,(1,pred_train_ratings.shape[0]))
    pred_test_ratings=np.reshape(pred_test_ratings.T,(1,pred_test_ratings.shape[0]))

    #--------evaluate-----------------MAE/RMSE-----------
    maeTrain[i] = mean_absolute_error(train_ratings, pred_train_ratings)
    maeTest[i] = mean_absolute_error(test_ratings, pred_test_ratings)
    
    rmseTrain[i]=math.sqrt(mean_squared_error(train_ratings, pred_train_ratings))
    rmseTest[i]=math.sqrt(mean_squared_error(test_ratings, pred_test_ratings))
    
    print("For k=", k, maeTest[i], rmseTest[i])
    
#----------Plots------------------------
plt.plot(ks, maeTest, 'k', label="Test")
plt.xlabel("k")
plt.ylabel("MAE")
plt.legend()
plt.title("MAE vs K for SVD")
plt.savefig("MAE-SVD.png")
plt.show()

plt.plot(ks, rmseTest, 'k', label="Test")
plt.xlabel("k")
plt.ylabel("RMSE")
plt.legend()
plt.title("RMSE vs K for SVD")
plt.savefig("RMSE-SVD.png")
plt.show()
