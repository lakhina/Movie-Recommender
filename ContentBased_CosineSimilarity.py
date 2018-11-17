# include utf-coding8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

# Function that get movie recommendations based on the cosine similarity score of movie genres and their year of release


def content_genre_year(user):
    debug = False
    flag = False
    movies_cols = []
    temp_genres = [i for i in range(5, 24)]
    movies_cols += temp_genres
    import pandas as pd
    # print movies_cols
    genre = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, usecols=movies_cols)

    title = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, usecols=[1])
    # Reading ratings file:
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols)

    df = ratings[ratings.rating == 5]

    df = df.groupby('user_id')['movie_id'].apply(list).reset_index(name='movie_ids')

    g_shape = genre.shape
    if debug:
        print(g_shape[0], g_shape[1])
    genre_list = genre.apply(lambda x: x.tolist(), axis=1)
    if debug:
        print(type(genre_list))
        print(genre_list.shape)
        print(genre_list.head())
        print(genre.as_matrix())

    titles = title[1]
    # genres = genre.as_matrix()
    genres = genre.values
    ind = [i for i in range(0, len(titles))]
    indices = pd.Series(ind, index=titles)
    title_by_id = pd.Series(titles, index=ind)
    movies_by_user = pd.Series(df.movie_ids.values, index=df.user_id)
    if debug:
        print movies_by_user
    # movie_title = 'Toy Story (1995)'
    # user = 1
    if user in df.user_id.values:
        movie_ids_selected = movies_by_user[user]
    else:
        movie_ids_selected = movies_by_user[1]
    # print movie_ids_selected
    ind_of_given_title = movie_ids_selected[0]
    print (
    "Since the user has rated the movie '", title_by_id[ind_of_given_title], "' as 5 , so we recommend these movies.")

    import json
    with open("genres_years.json") as fp:
        genres = json.load(fp)
    # print numpy.array(genres)[:10]
    import pandas as pd
    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, usecols=[1])
    # print movies.head()
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0)
    tfidf_matrix = tf.fit_transform(genres)

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    titles = movies[1]
    indices = pd.Series(movies.index, index=movies[1])
    # print "INDICES-------------------"
    # print indices
    # title = 'Taxi Driver (1976)'
    # print('Input: ' + title)
    # idx = indices[title]
    idx=ind_of_given_title
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:8]
    movie_indices = [i[0] for i in sim_scores]
    # print movie_indices
    # movie_recommendations = titles.iloc[movie_indices]
    #
    # print(movie_recommendations)
    movie_recommendations=[]
    for id in movie_indices:
        print title_by_id[id]
        if id!=idx:
            movie_recommendations.append(title_by_id[id])
    return movie_recommendations
content_genre_year(3)

