import pandas as pd
from scipy import spatial

movies_cols = []
temp_genres = [i for i in range(5, 24)]
movies_cols += temp_genres
# movies_cols=["id","title"]
# temp_genres=[dict_genres[i-5] for i in range(5,24)]
# movies_cols+=temp_genres
print movies_cols
genre = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, usecols=movies_cols)
print genre.head()
title = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, usecols=[1])
# print title.head()
print genre.shape
g_shape = genre.shape
print g_shape[0], g_shape[1]
genre_list = genre.apply(lambda x: x.tolist(), axis=1)
print type(genre_list)
print genre_list.shape
print genre_list.head()
print genre.as_matrix()
# genre_arr=np.array(genre).reshape(g_shape[0],1)
# print genre_arr.shape
titles = title[1]
genres = genre.as_matrix()
ind = [i for i in range(0, len(titles))]
indices = pd.Series(ind, index=titles)
title_by_id = pd.Series(titles, index=ind)
# indices_genre_by_id = pd.Series(genres, index=ind)
# print indices_genre_by_id
title = 'Good Will Hunting (1997)'
ind_of_given_title = indices[title]

print('Input: ' + title)
print ind_of_given_title
genre_of_inp_title = genres[ind_of_given_title]
cosine_similarity_dict = {}
for i in range(0, len(titles)):
    cosine_sim = 1 - spatial.distance.cosine(genre_of_inp_title, genres[i])
    cosine_similarity_dict[i] = cosine_sim

print cosine_similarity_dict
sorted_cos_sim_movie_id_pairs = sorted([(v, k) for (k, v) in cosine_similarity_dict.items()], reverse=True)
top_selected = sorted_cos_sim_movie_id_pairs[:20]
print top_selected
top_selected_movie_ids = [top_selected[i][1] for i in range(len(top_selected))]
print top_selected_movie_ids
print "Movie Recommendations :"
for id in top_selected_movie_ids:
    print title_by_id[id]
