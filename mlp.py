# include utf-coding8
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier


def mlp_recommender(userid):
    col_names = ['user_id', 'item_id', 'rating', 'timestamp']
    col_names1 = ['userno', 'age', 'gender', 'occu', 'timestamp']
    col_names2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                  'u', 'v', 'w', 'x']

    # reading csv file to calculate the unique users and items
    data = pd.read_csv('ml-100k/u.data', sep='\t', names=col_names, encoding='latin-1')
    users = data.user_id.unique().shape[0]  # no of users
    items = data.item_id.unique().shape[0]  # no of items

    # reading the five fold data available
    traina = pd.read_csv('ml-100k/ua.base', sep='\t', names=col_names, encoding='latin-1')
    testa = pd.read_csv('ml-100k/ua.test', sep='\t', names=col_names, encoding='latin-1')
    trainb = pd.read_csv('ml-100k/ub.base', sep='\t', names=col_names, encoding='latin-1')
    testb = pd.read_csv('ml-100k/ub.test', sep='\t', names=col_names, encoding='latin-1')
    train1 = pd.read_csv('ml-100k/u1.base', sep='\t', names=col_names, encoding='latin-1')
    test1 = pd.read_csv('ml-100k/u1.test', sep='\t', names=col_names, encoding='latin-1')
    train2 = pd.read_csv('ml-100k/u2.base', sep='\t', names=col_names, encoding='latin-1')
    test2 = pd.read_csv('ml-100k/u2.test', sep='\t', names=col_names, encoding='latin-1')
    train3 = pd.read_csv('ml-100k/u3.base', sep='\t', names=col_names, encoding='latin-1')
    test3 = pd.read_csv('ml-100k/u3.test', sep='\t', names=col_names, encoding='latin-1')
    train4 = pd.read_csv('ml-100k/u4.base', sep='\t', names=col_names, encoding='latin-1')
    test4 = pd.read_csv('ml-100k/u4.test', sep='\t', names=col_names, encoding='latin-1')
    train5 = pd.read_csv('ml-100k/u5.base', sep='\t', names=col_names, encoding='latin-1')
    test5 = pd.read_csv('ml-100k/u5.test', sep='\t', names=col_names, encoding='latin-1')
    user_info = pd.read_csv('ml-100k/u.user', sep='|', names=col_names1, encoding='latin-1')
    item_info = pd.read_csv('ml-100k/u.item', sep='|', names=col_names2, encoding='latin-1')
    user_info = np.array(user_info)
    item_info = np.array(item_info)
    # print (item_info)


    '''
    0
    1 age
    2 gender
    3 occu
    
    
    
    
    '''

    ##print user_info.shape
    # print (user_info)
    train_ratings = []
    train_features = []
    test_features = []
    ages = []

    ratings = []
    for i in range(len(user_info)):
        ages.append(user_info[i][1])

    # print (np.min(ages))
    # print (np.max(ages))

    def transform_ratings(r):
        for i in range(0, len(r)):
            if r[i] > 3:
                r[i] = 3
            elif r[i] == 3:
                r[i] = 2
            else:
                r[i] = 1
        return r

    # extracting item meta data for preparing test data
    def genretest(i, count):
        for k in range(5, 24):
            test_features[count].append(item_info[i][k])

    # extracting user meta data
    # gender information
    def gendertest(i, count):
        # males
        if user_info[i][2] == 'M':
            test_features[count].extend([1, 0])
        # females
        else:
            test_features[count].extend([0, 1])

    # age information extraction
    def agetest(i, count):
        if user_info[i][1] >= 7:
            if user_info[i][1] <= 12:
                test_features[count].extend([1, 0, 0, 0, 0, 0, 0])
        if user_info[i][1] >= 12:
            if user_info[i][1] <= 19:
                test_features[count].extend([0, 1, 0, 0, 0, 0, 0])

        if user_info[i][1] >= 20:
            if user_info[i][1] <= 30:
                test_features[count].extend([0, 0, 1, 0, 0, 0, 0])

        if user_info[i][1] >= 31:
            if user_info[i][1] <= 40:
                test_features[count].extend([0, 0, 0, 1, 0, 0, 0])

        if user_info[i][1] >= 41:
            if user_info[i][1] <= 50:
                test_features[count].extend([0, 0, 0, 0, 1, 0, 0])

        if user_info[i][1] >= 51:
            if user_info[i][1] <= 60:
                test_features[count].extend([0, 0, 0, 0, 0, 1, 0])

        if user_info[i][1] >= 61:
            if user_info[i][1] <= 77:
                test_features[count].extend([0, 0, 0, 0, 0, 0, 1])

    # occupation information
    def occupationtest(i, count):
        if user_info[i][3] == "administrator":
            test_features[count].extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "artist":
            test_features[count].extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "doctor":
            test_features[count].extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "educator":
            test_features[count].extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "engineer":
            test_features[count].extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "entertainment":
            test_features[count].extend([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "executive":
            test_features[count].extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "healthcare":
            test_features[count].extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "homemaker":
            test_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "lawyer":
            test_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "librarian":
            test_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "marketing":
            test_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "none":
            test_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "other":
            test_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "programmer":
            test_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "retired":
            test_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "salesman":
            test_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        elif user_info[i][3] == "scientist":
            test_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif user_info[i][3] == "student":
            test_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif user_info[i][3] == "technician":
            test_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        elif user_info[i][3] == "writer":
            test_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    # test ratings given in test files
    ratings = []

    # preparing meta data to be fed as input
    def metadata(train_matrix, test_matrix):
        ratings_train = []
        ratings = []
        count = 0
        for i in range(0, 943):
            for j in range(0, 1682):
                if train_matrix[i][j] != 0:
                    train_features.append([])
                    gender(i, count)
                    occupation(i, count)
                    age(i, count)
                    genre(j, count)
                    ratings_train.append(int(train_matrix[i][j]))
                    count += 1
        count = 0
        for i in range(0, 943):
            for j in range(0, 1682):
                if test_matrix[i][j] != 0:
                    test_features.append([])
                    gendertest(i, count)
                    occupationtest(i, count)
                    agetest(i, count)
                    genretest(j, count)
                    ratings.append(int(test_matrix[i][j]))
                    count += 1

        '''     
        tm=train_matrix.T
        for i in range (0,1682):
            for j in range(0,943):
                if tm[i][j]!=0:
                    genre(i)
        '''
        ##print len(train_features)
        '''
        for i in range (0,943):
            for j in range(0,1682):
                if test_matrix[i][j]!=0:
                    gendertest(i)
                    #occupationtest(i)
                    #agetest(i)
            
        #in order to append item features
        tesm=test_matrix.T
        for i in range (0,1682):
            for j in range(0,943):
                if tesm[i][j]!=0:
                    genretest(i)
         '''
        return train_features, test_features, ratings_train, ratings

    # constructing the test and train matrix

    def A(train_data, test_data, num):
        train_matrix = np.zeros((users, items))
        test_matrix = np.zeros((users, items))

        for row in train_data.itertuples():  # user-item matrix provided with the corresponding ratings
            train_matrix[row[1] - 1, row[2] - 1] = row[3]  # train data matrix accessing rowwise
        for row in test_data.itertuples():
            test_matrix[row[1] - 1, row[2] - 1] = row[3]  # test data matrix
        # print (train_matrix)
        # train_matrix,test_matrix=normalize_data(train_matrix,test_matrix)
        for i in range(0, 943):
            for j in range(0, 1682):
                if train_matrix[i][j] != 0:
                    #	#print (i,j)
                    train_ratings.append(train_matrix[i][j])
        # print (train_matrix)
        # print (len(train_ratings))
        ##print test_matrix.shape
        return train_matrix, test_matrix, train_ratings

    def gender(i, count):
        if user_info[i][2] == 'M':
            train_features[count].extend([1, 0])

        else:
            train_features[count].extend([0, 1])

    def age(i, count):
        if user_info[i][1] >= 7:
            if user_info[i][1] <= 12:
                train_features[count].extend([1, 0, 0, 0, 0, 0, 0])
        if user_info[i][1] >= 12:
            if user_info[i][1] <= 19:
                train_features[count].extend([0, 1, 0, 0, 0, 0, 0])

        if user_info[i][1] >= 20:
            if user_info[i][1] <= 30:
                train_features[count].extend([0, 0, 1, 0, 0, 0, 0])

        if user_info[i][1] >= 31:
            if user_info[i][1] <= 40:
                train_features[count].extend([0, 0, 0, 1, 0, 0, 0])

        if user_info[i][1] >= 41:
            if user_info[i][1] <= 50:
                train_features[count].extend([0, 0, 0, 0, 1, 0, 0])

        if user_info[i][1] >= 51:
            if user_info[i][1] <= 60:
                train_features[count].extend([0, 0, 0, 0, 0, 1, 0])

        if user_info[i][1] >= 61:
            if user_info[i][1] <= 77:
                train_features[count].extend([0, 0, 0, 0, 0, 0, 1])

    # occupation information

    # extracted from u.user
    def occupation(i, count):
        if user_info[i][3] == "administrator":
            train_features[count].extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "artist":
            train_features[count].extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "doctor":
            train_features[count].extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "educator":
            train_features[count].extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "engineer":
            train_features[count].extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "entertainment":
            train_features[count].extend([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "executive":
            train_features[count].extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "healthcare":
            train_features[count].extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "homemaker":
            train_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "lawyer":
            train_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "librarian":
            train_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "marketing":
            train_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "none":
            train_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "other":
            train_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "programmer":
            train_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "retired":
            train_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif user_info[i][3] == "salesman":
            train_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        elif user_info[i][3] == "scientist":
            train_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif user_info[i][3] == "student":
            train_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif user_info[i][3] == "technician":
            train_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        elif user_info[i][3] == "writer":
            train_features[count].extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    def genre(i, count):
        for k in range(5, 24):
            train_features[count].append(item_info[i][k])

    # fold 1
    if True:
        train_matrix, test_matrix, ratings_train = A(train2, test2, 1)
        ratings_train = np.array(ratings_train)

        for i in range(0, 943):
            for j in range(0, 1682):
                if test_matrix[i][j] != 0:
                    ratings.append(int(test_matrix[i][j]))
                    # ratings=transform_ratings(ratings)

        # this function implements the improvised version of gaussian nb
        # ratings_train=transform_ratings(ratings_train)
        # print ((train_features))
        train_features, test_features, ratings_train, ratings = metadata(train_matrix, test_matrix)
        # train_features,test_features,ratings,dimension=folds(train_features,test_features)
        train_features = np.asarray(train_features)
        # print (train_features)
        test_features = np.asarray(test_features)
        # print (test_features.shape)
        # print (test_features)
        ##print ("-- ",np.asarray(test_features).shape)
        # print (" --- ",ratings_train)
        '''
        temp=ratings_train
        ratings_train=[]
        for i in range(0,len(temp)):
            tp=[]
            tp.append(temp[i])
            ratings_train.append(tp)
        
        '''
        # print (" --- ",ratings_train)

        prediction = []
        prediction = []
        ##print ((train_features))

        ratings = transform_ratings(ratings)
        ratings_train = transform_ratings(ratings_train)


        gnb = MLPClassifier()

        clf = gnb.fit(train_features, ratings_train)
        joblib.dump(clf, 'mlp.pkl')
        pre = clf.predict(test_features)


        # sklearn.metrics.mean_squared_error


        print ("MAE :", mean_absolute_error(pre, ratings))
        print ("RMSE :", mean_squared_error(ratings, pre))

        from sklearn.metrics import accuracy_score

        print ("Accuracy : ", accuracy_score(ratings, pre))

        cnf_matrix = confusion_matrix(pre, ratings)
        # Plot non-normalized confusion matrix
        plt.figure()
        # if ratings are not transformed classes=[1,2,3,4,5]


        # RECOMMENDATION

        # userid=6


        test_features = []
        i = userid
        for j in range(0, 1682):
            count = j
            ##print (" ----- ",count)
            test_features.append([])
            ##print (type(test_features[0]))
            gendertest(i, count)
            occupationtest(i, count)
            agetest(i, count)
            genretest(j, count)

        result = []
        result = clf.predict(test_features)
        # print (result)
        # print (result.shape)
        final = np.asarray(result)
        recommended = final.argsort()[-5:][::-1]

        # test_features=[]

        title = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, usecols=[1])
        titles = title[1]
        ind = [i for i in range(0, len(titles))]
        indices = pd.Series(ind, index=titles)
        title_by_id = pd.Series(titles, index=ind)

        top_selected_movie_ids = recommended
        # print(top_selected_movie_ids)
        result = []
        for id in top_selected_movie_ids:
            # #print(title_by_id[id])
            result.append(title_by_id[id])
        # print (result)
        return result

userid = 6

print "------------------------------------------"
print "NEURAL NETWORK :"
movie_titles = mlp_recommender(userid)
print "Movie Recommendations  for userid ",userid," :"
for m in movie_titles:
    print m
