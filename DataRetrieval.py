
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
