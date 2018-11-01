"""data Retrieval"""
import pandas as pd

ratings = pd.read_csv('ml-100k/u.data', sep='\t', encoding='latin-1', header=None, usecols=[0, 1, 2])
print ratings.head()
print ratings.shape
genres = pd.read_csv('ml-100k/u.genre', sep='|', encoding='latin-1', header=None)
print genres.head()
dict_genres = dict(zip(genres[1], genres[0]))
print dict_genres
movies_cols = [0, 1]
temp_genres = [i for i in range(5, 24)]
movies_cols += temp_genres

print movies_cols
movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, usecols=movies_cols)
print movies.head()

import numpy

movies_arr = numpy.array(movies)
print "Movies array----------------------"
print movies_arr.shape
print movies_arr.T[0]
for i in range(2, 21):
    for j in range(len(movies_arr.T[0])):
        if movies_arr.T[i][j] == 1:
            movies_arr.T[i][j] = dict_genres[i - 2]
        if movies_arr.T[i][j] == 0:
            movies_arr.T[i][j] = ""

print movies_arr.T[5:7]
print movies_arr[:10]
print movies_arr.shape
print len(movies_arr[0])
movies_arr_cleaned = []
import re

temp = ""
for i in range(len(movies_arr.T[0])):
    for j in range(2, 21):
        movies_arr[i][j] = ''.join([k if ord(k) < 128 else ' ' for k in movies_arr[i][j]])
        temp += str(movies_arr[i][j]).encode('utf-8') + " "
        temp.encode('utf-8')
        temp = re.sub("\s+", ' ', temp)

    movies_arr_cleaned.append(temp)
    temp = ""

import json

with open("genres.json", 'w') as fp:
    json.dump(movies_arr_cleaned, fp)
print len(movies_arr_cleaned)
df = pd.DataFrame(movies_arr_cleaned)
df.to_csv("genres.csv")
