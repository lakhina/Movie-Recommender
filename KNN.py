#include utf-coding8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

col_names= ['user_id', 'item_id', 'rating', 'timestamp']
col_names1=['userno','age','gender','occu','timestamp']
col_names2=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x']

#reading csv file to calculate the unique users and items
data = pd.read_csv('ml-100k/u.data', sep='\t', names=col_names)          
users = data.user_id.unique().shape[0]           #no of users
items = data.item_id.unique().shape[0]           #no of items

#reading the five fold data available 
traina=pd.read_csv('ml-100k/ua.base',sep='\t',names=col_names)            
testa=pd.read_csv('ml-100k/ua.test',sep='\t',names=col_names)
trainb=pd.read_csv('ml-100k/ub.base',sep='\t',names=col_names)            
testb=pd.read_csv('ml-100k/ub.test',sep='\t',names=col_names)
train1=pd.read_csv('ml-100k/u1.base',sep='\t',names=col_names)            
test1=pd.read_csv('ml-100k/u1.test',sep='\t',names=col_names)
train2=pd.read_csv('ml-100k/u2.base',sep='\t',names=col_names)
test2=pd.read_csv('ml-100k/u2.test',sep='\t',names=col_names)
train3=pd.read_csv('ml-100k/u3.base',sep='\t',names=col_names)
test3=pd.read_csv('ml-100k/u3.test',sep='\t',names=col_names)
train4=pd.read_csv('ml-100k/u4.base',sep='\t',names=col_names)
test4=pd.read_csv('ml-100k/u4.test',sep='\t',names=col_names)
train5=pd.read_csv('ml-100k/u5.base',sep='\t',names=col_names)
test5=pd.read_csv('ml-100k/u5.test',sep='\t',names=col_names)
user_info=pd.read_csv('ml-100k/u.user',sep='|',names=col_names1)
item_info=pd.read_csv('ml-100k/u.item',sep='|',names=col_names2)
user_info=np.array(user_info)
item_info=np.array(item_info)
print("Item info :\n", item_info)
#print user_info.shape
print("User info :\n", user_info)
train_ratings=[]
train_features=[]
test_features=[]
ages=[]
for i in range (len(user_info)):
    ages.append(user_info[i][1])
print("Min ages :", np.min(ages))
print("Max ages :", np.max(ages))

def transform_ratings(r):
    for i in range (0,len(r)):
        if r[i]>3:
            r[i]=2
        elif r[i]==3:
            r[i]=1
        else :
            r[i]=0
    return r

#extracting item meta data for preparing test data
def genretest(i):
    for k in range (5,24):
        test_features.append( item_info[i][k])

#extracting user meta data
#gender information
def gendertest(i):
    #males
    if user_info[i][2]=='M':
        test_features.extend([1,0])
    #females
    else:
        test_features.extend([0,1])

#age information extraction
def agetest(i):
    if user_info[i][1] >=7 :
        if user_info[i][1] <17:
            test_features.extend([1,0,0,0,0,0,0])
    if user_info[i][1] >=17:
        if user_info[i][1] <27:
            test_features.extend([0,1,0,0,0,0,0])

    if user_info[i][1] >=27:
        if user_info[i][1] <37:
            test_features.extend([0,0,1,0,0,0,0])

    if user_info[i][1] >=37:
        if user_info[i][1] <47:
            test_features.extend([0,0,0,1,0,0,0])

    if user_info[i][1] >=47:
        if user_info[i][1] <57:
            test_features.extend([0,0,0,0,1,0,0])

    if user_info[i][1] >=57:
        if user_info[i][1] <67:
            test_features.extend([0,0,0,0,0,1,0])

    if user_info[i][1] >=67 :
        if user_info[i][1] <77:
            test_features.extend([0,0,0,0,0,0,1])

#occupation information
def occupationtest(i):
    if user_info[i][3]=="administrator":
        test_features.extend([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="artist":
        test_features.extend([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="doctor":
        test_features.extend([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="educator":
        test_features.extend([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="engineer":
        test_features.extend([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="entertainment":
        test_features.extend([1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="executive":
        test_features.extend([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="healthcare":
        test_features.extend([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="homemaker":
        test_features.extend([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="lawyer":
        test_features.extend([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="librarian":
        test_features.extend([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="marketing":
        test_features.extend([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="none":
        test_features.extend([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="other":
        test_features.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
    elif user_info[i][3]=="programmer":
        test_features.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
    elif user_info[i][3]=="retired":
        test_features.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
    elif user_info[i][3]=="salesman":
        test_features.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
    elif user_info[i][3]=="scientist":
        test_features.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
    elif user_info[i][3]=="student":
        test_features.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
    elif user_info[i][3]=="technician":
        test_features.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
    elif user_info[i][3]=="writer":
        test_features.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])


def confmatrix(c_matrix, ratings):

    normalize=False
    length = np.arange(len(ratings))
    plt.imshow(c_matrix, cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    print ("Confusion matrix :",c_matrix)
    plt.xticks(length, ratings)
    plt.yticks(length, ratings)
    plt.xlabel('Predicted labels')
    plt.ylabel('Observed labels')
    plt.show()

def fitting(m1,m2,samples):
    plt.plot(samples,m1,label='testing set')
    plt.plot(samples,m2,label='training set')
    plt.title("overfitting/underfitting")
    plt.legend()
    plt.show()

#preparing the data 
def folds(train_features,test_features):
    train_features=np.array(train_features)
        #print len(train_features)
    dimension=len(train_features)/21
    train_features=train_features.reshape((dimension,21))
        #print train_features.shape
    test_features=np.array(test_features)
    test_features=test_features.reshape((len(test_features)/21,21))
    print("Test features shape :", test_features.shape)

#test ratings given in test files
    ratings=[]
    for i in range(0,943):
        for j in range (0,1682):
            if test_matrix[i][j]!=0:
                ratings.append(test_matrix[i][j])
    ratings=transform_ratings(ratings)
    return train_features,test_features,ratings,dimension


def calculate(classifier):
    mae1=[]
    mae2=[]
    num_samples=[80000,82500,85000,88000,90569]

    for n in num_samples:
        model=classifier.fit(train_features[n:,:],ratings_train[n:])
        pred1=model.predict(test_features)
        pred2=model.predict(train_features)
        mae1.append(mean_absolute_error(pred1,ratings))
        mae2.append(mean_absolute_error(pred2,ratings_train))

    fitting(mae1,mae2,num_samples)


#preparing meta data to be fed as input
def metadata(train_matrix,test_matrix):

    for i in range (0,943):
        for j in range(0,1682):
            if train_matrix[i][j] !=0:
                gender(i)
                # occupation(i)
                # age(i)
    #print count1
        
    tm=train_matrix.T
    for i in range (0,1682):
        for j in range(0,943):
            if tm[i][j]!=0:
                genre(i)
       
    #print len(train_features)

    for i in range (0,943):
        for j in range(0,1682):
            if test_matrix[i][j]!=0:
                gendertest(i)
                # occupationtest(i)
                # agetest(i)
        
    #in order to append item features
    tesm=test_matrix.T
    for i in range (0,1682):
        for j in range(0,943):
            if tesm[i][j]!=0:
                genretest(i)

    return train_features,test_features

#constructing the test and train matrix

def A(train_data,test_data,num):
    train_matrix = np.zeros((users, items))
    test_matrix=np.zeros((users,items))

    for row in train_data.itertuples(): #user-item matrix provided with the corresponding ratings
        train_matrix[row[1]-1, row[2]-1] = row[3]     #train data matrix accessing rowwise
    for row in test_data.itertuples():
        test_matrix[row[1]-1, row[2]-1]=row[3]#test data matrix

    print("Train Matrix : \n", train_matrix, "\n", train_matrix.shape)
    #train_matrix,test_matrix=normalize_data(train_matrix,test_matrix)
    for i in range(0,943):
        for j in range (0,1682):
            if train_matrix[i][j]!=0:
            #	print i,j
                train_ratings.append(train_matrix[i][j])
    print("Train Matrix : \n", train_matrix, "\n", train_matrix.shape)
    print("Len train ratings :", len(train_ratings))
    #print test_matrix.shape
    return train_matrix,test_matrix,train_ratings

def gender(i):
    if user_info[i][2]=='M':
        train_features.extend([1,0])

    else:
        train_features.extend([0,1])

def age(i):
    if user_info[i][1] >=7 :
        if user_info[i][1] <17:
            train_features.extend([1,0,0,0,0,0,0])
    if user_info[i][1] >=17:
        if user_info[i][1] <27:
            train_features.extend([0,1,0,0,0,0,0])

    if user_info[i][1] >=27:
        if user_info[i][1] <37:
            train_features.extend([0,0,1,0,0,0,0])

    if user_info[i][1] >=37:
        if user_info[i][1] <47:
            train_features.extend([0,0,0,1,0,0,0])

    if user_info[i][1] >=47:
        if user_info[i][1] <57:
            train_features.extend([0,0,0,0,1,0,0])

    if user_info[i][1] >=57:
        if user_info[i][1] <67:
            train_features.extend([0,0,0,0,0,1,0])

    if user_info[i][1] >=67 :
        if user_info[i][1] <77:
            train_features.extend([0,0,0,0,0,0,1])



#extracted from u.user
def occupation(i):
    if user_info[i][3]=="administrator":
        train_features.extend([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="artist":
        train_features.extend([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="doctor":
        train_features.extend([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="educator":
        train_features.extend([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="engineer":
        train_features.extend([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="entertainment":
        train_features.extend([1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="executive":
        train_features.extend([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="healthcare":
        train_features.extend([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="homemaker":
        train_features.extend([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="lawyer":
        train_features.extend([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="librarian":
        train_features.extend([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="marketing":
        train_features.extend([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="none":
        train_features.extend([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
    elif user_info[i][3]=="other":
        train_features.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
    elif user_info[i][3]=="programmer":
        train_features.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
    elif user_info[i][3]=="retired":
        train_features.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
    elif user_info[i][3]=="salesman":
        train_features.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
    elif user_info[i][3]=="scientist":
        train_features.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
    elif user_info[i][3]=="student":
        train_features.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
    elif user_info[i][3]=="technician":
        train_features.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
    elif user_info[i][3]=="writer":
        train_features.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])

def genre(i):
    for k in range (5,24):
        train_features.append( item_info[i][k])
        # print "in genre \n",train_features.__len__()
#fold 1 
train_matrix,test_matrix,ratings_train=A(train2,test2,1)
print("check 1 ", train_matrix.shape)
print("check 2 ", test_matrix.shape)
ratings_train=np.array(ratings_train)

# this function implements the improvised version of knn
ratings_train=transform_ratings(ratings_train)
train_features,test_features=metadata(train_matrix,test_matrix)
print("test 1", np.array(train_features).shape)
train_features,test_features,ratings,dimension=folds(train_features,test_features)
print("test 2", np.array(train_features).shape)
print(type(train_features))
print(dimension)
import pandas as pd
df = pd.DataFrame(train_features)
df.to_csv("train.csv")
df = pd.DataFrame(test_features)
df.to_csv("test.csv")
df = pd.DataFrame(ratings)
df.to_csv("ratings.csv")
# df = pd.DataFrame(dimension)
# df.to_csv("dimension.csv")

print("Train features shape :", train_features.shape)
print("Test features shape :", test_features.shape)
prediction=[]

# classifier=GaussianNB()
classifier=neighbors.KNeighborsClassifier()
clf=classifier.fit(train_features,ratings_train)

joblib.dump(clf, 'KNN.pkl')
pre=clf.predict(test_features)

print("Mean absolute error :", mean_absolute_error(pre, ratings))

print("Mean squared error :", mean_squared_error(pre, ratings))
if dimension == 19:
    calculate(classifier)
#print mae1,rmse1
#print mae2,rmse2
cnf_matrix=confusion_matrix(pre,ratings)
# Plot non-normalized confusion matrix
plt.figure()
#if ratings are not transformed classes=[1,2,3,4,5]
cl1=[1,2,3]
cl2=[1,2,3,4,5]
confmatrix(cnf_matrix, cl1)















