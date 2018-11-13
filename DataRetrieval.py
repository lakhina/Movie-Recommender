# include utf-coding8
# import numpy as np
# import csv
# import sys
# import os
# from os.path import join
# import copy
# # import DropboxAPI
#
#
# #Fetching DataSet from DropBox  and Unzipping the File
# # url ='https://www.dropbox.com/sh/euppz607r6gsen2/AAAQCu8KjT7Ii1R60W2-Bm1Ua/MovieLens%20(Movie%20Ratings)?dl=1'
# # zipFileName = 'MovieLens (Movie Ratings).zip'
# # subzipFileName ='movielens100k/ml-100k'
# userDataSet = 'u.data'
# userTestDataSet = 'u1.test'
# # destPath = os.getcwd()
# # DropboxAPI.fetchData(url, zipFileName, destPath)
# # filePath = join(destPath, zipFileName.rsplit(".", 1)[0])
# # filePath = join(filePath,subzipFileName.rsplit(".", 1)[0])
# # fullFilePath = join(filePath,userDataSet)
# fullFilePath="ml-100k/u.data"
# #Importing the Dataset
# csvfile = open(fullFilePath)
# csvreader = csv.reader(csvfile, delimiter='\t')
#
# data=[]
# for row in csvreader:
#     data.append(row)
# csvfile.close()
# testid=np.random.randint(1,100000, 10000)
# testset=[]
# k=1
# for i in testid:
#     i-=k
#     k+=1
#     testset.append(data[i])
# #print('Select test 10000 cases.')
# mdata=np.array(data,dtype=int)
# usernum=int(np.max(mdata[:,0]))
# itemnum=int(np.max(mdata[:,1]))
# #print('Total user number is : '+str(usernum))
# #print('Total movie number is : '+str(itemnum))
# fdata = np.array(np.zeros((usernum, itemnum)))
# #print('Formating matrix.')
# for row in mdata:
#     fdata[row[0]-1, row[1]-1] = row[2]
# for case in testset:
#     fdata[int(case[0])-1,int(case[1])-1]=0
#
# def findKNNitem(indata, item):
#     iid = int(item[1])
#     uid = int(item[0])
#     temp = copy.deepcopy(indata[:, iid-1])
#     for j in range(itemnum):
#         indata[:, j] -= temp
#     indata = indata**3
#     sumd=indata.sum(axis=0)
#     max=sumd.max()
#     nn=[]
#     for l in range(5):
#         while fdata[uid-1,sumd.argmin()] == 0:
#             sumd[sumd.argmin()]=max
#         nn.append(sumd.argmin())
#     ratelist = []
#     for j in range(5):
#         ratelist.append(fdata[uid-1, nn[j]])
#     rate = np.average(ratelist)
#     error = np.absolute(int(item[2])-rate)
#     return error
#
#
# def findKNNuser(indata,item):
#     iid = int(item[1])
#     uid = int(item[0])
#     temp = copy.deepcopy(indata[uid-1, :])
#     for i in range(usernum):
#         indata[i, :] -= temp
#     indata = indata**3
#     sumd=indata.sum(axis=1)
#     max=sumd.max()
#     nn=[]
#     z=5
#     for i in range(z):
#         while fdata[sumd.argmin(),iid-1]==0:
#             sumd[sumd.argmin()]=max
#             if sumd.max()==sumd.min():
#                 z=i
#                 break
#
#         nn.append(sumd.argmin())
#     ratelist=[]
#     for i in range(z):
#         ratelist.append(fdata[nn[i], iid-1])
#     rate = np.average(ratelist)
#     error=np.absolute(int(item[2])-rate)
#     return error
#
# errorlist=[]
# #print('start test '+str(len(testset)))
# n=1
#
# for testcase in testset:
#     if n % 100==0:
#         #print('Test has finished : '+str(n/100)+'%' )
#     error1=findKNNuser(copy.deepcopy(fdata), testcase)
#     errorlist.append((error1)**2)
#     n+=1
# #print('User-based KNN - Test finished')
# meanserror=np.average(errorlist)
# #print('User-based MSE: ', meanserror)
#
# for testcase in testset:
#     if n % 100==0:
#         #print('Test has finished : '+str(n/100)+'%' )
#     error2=findKNNitem(copy.deepcopy(fdata), testcase)
#     errorlist.append((error2)**2)
#     n+=1
# #print('test finished')
# meanserror=np.average(errorlist)
# #print('Item-based MSE: ', meanserror)
#
# for testcase in testset:
#     if n % 100==0:
#         #print('Test has finished : '+str(n/100)+'%' )
#     error1=findKNNuser(copy.deepcopy(fdata), testcase)
#     error2=findKNNitem(copy.deepcopy(fdata), testcase)
#     errorlist.append((error1/2+error2/2)**2)
#     n+=1
# #print('test finished')
# meanserror=np.average(errorlist)
# #print('Mixed User-Item based: ',meanserror)


"""data Retrieval"""
import pandas as pd

ratings = pd.read_csv('ml-100k/u.data', sep='\t', encoding='latin-1', header=None, usecols=[0, 1, 2])
# print ratings.head()
# print ratings.shape
genres = pd.read_csv('ml-100k/u.genre', sep='|', encoding='latin-1', header=None)
#print genres.head()
dict_genres = dict(zip(genres[1], genres[0]))
#print dict_genres
# movies = pd.read_csv('data/movies.csv', sep='\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])
movies_cols = [0, 1]
temp_genres = [i for i in range(5, 24)]
movies_cols += temp_genres
# movies_cols=["id","title"]
# temp_genres=[dict_genres[i-5] for i in range(5,24)]
# movies_cols+=temp_genres
#print movies_cols
movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, usecols=movies_cols)
#print movies.head()

# for i in range(5,24):
#     #print movies.loc[(movies[i] ==1)]
# movies.loc[(movies[i] ==1)] = dict_genres[i-5]
import numpy

genres_flag = False
years_flag = True
if genres_flag:
    movies_arr = numpy.array(movies)
    # print "Movies array----------------------"
    # print movies_arr.shape
    # print movies_arr.T[0]
    for i in range(2, 21):
        for j in range(len(movies_arr.T[0])):
            if movies_arr.T[i][j] == 1:
                movies_arr.T[i][j] = dict_genres[i - 2]
            if movies_arr.T[i][j] == 0:
                movies_arr.T[i][j] = ""

    # print movies_arr.T[5:7]
    # print movies_arr[:10]
    # print movies_arr.shape
    # print len(movies_arr[0])
    movies_arr_cleaned = []
    import re

    temp = ""
    for i in range(len(movies_arr.T[0])):
        for j in range(2, 21):
            movies_arr[i][j] = ''.join([k if ord(k) < 128 else ' ' for k in movies_arr[i][j]])
            temp += str(movies_arr[i][j]).encode('utf-8') + " "
            temp.encode('utf-8')
            temp = re.sub("\s+", ' ', temp)
        # #print temp
        movies_arr_cleaned.append(temp)
        temp = ""
    # #print movies.head()
    # #print movies.shape
    import json

    with open("genres.json", 'w') as fp:
        json.dump(movies_arr_cleaned, fp)
    # print len(movies_arr_cleaned)
    df = pd.DataFrame(movies_arr_cleaned)
    df.to_csv("genres.csv")
import re

if years_flag:
    movies_arr = numpy.array(movies)
    print ("Movies array----------------------")
    print (movies_arr.shape)
    print(movies_arr.T[1])
    movies_dict = {}


    def get_year(name):
        search_res = re.search('\s\((\d{4})\)', name)
        if search_res != None:
            return search_res.groups()[0]
        else:
            return name


    movies_arr_cleaned_years = []
    for name in movies_arr.T[1]:
        if name != None and type(name) != None:
            # print(get_year(name),name)
            if get_year(name) in movies_dict.keys():
                temp = movies_dict[get_year(name)]
                temp.append(name)
                movies_dict[get_year(name)] = temp
            else:
                temp = []
                temp.append(name)
                movies_dict[get_year(name)] = temp
            movies_arr_cleaned_years.append(get_year(name))
        else:
            # print(name)
            movies_arr_cleaned_years.append(name)

    import json

    with open("years.json", 'w')as fp:
        json.dump(movies_arr_cleaned_years, fp)
    with open("movie_years.json", 'w')as fp:
        json.dump(movies_dict, fp)
    df1 = pd.DataFrame(movies_arr_cleaned_years)
    df1.to_csv("years.csv")

    with open("genres.json") as fp:
        genres = json.load(fp)
    print(genres)
    finalarr = []
    for i in range(len(genres)):
        temp = genres[i] + movies_arr_cleaned_years[i]
        print(temp)
        finalarr.append(temp)
        temp = ""
    print(finalarr)
    with open("genres_years.json", 'w')as fp:
        json.dump(finalarr, fp)
    df1 = pd.DataFrame(finalarr)
    df1.to_csv("genres_years.csv")
    # print len(movies_arr_cleaned)
