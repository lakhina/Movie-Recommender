from flask import Flask, render_template, request

from cf import getRecommendation
from cf import getRecommended

from ContentBased_CosineSimilarity import content_genre_year
app = Flask(__name__)
import numpy as np


def last(n):
    return n[-1]


def recommendSVD(userID, count):
    user = userID - 1
    import pandas as pd
    title = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, usecols=[1])
    titles = title[1]
    ind = [i for i in range(0, len(titles))]
    indices = pd.Series(ind, index=titles)
    title_by_id = pd.Series(titles, index=ind)
    AlreadymovieList = []
    pred = np.loadtxt('modelSVD1', delimiter=',')
    train_rows = np.loadtxt('modelSVD2', delimiter=',')
    train_cols = np.loadtxt('modelSVD3', delimiter=',')
    userRatings = pred[user]
    for idx, val in enumerate(train_rows):
        if val == user:
            AlreadymovieList.append(train_cols[idx])
    Movie_Ratgs = []
    for idx, val in enumerate(userRatings):
        if not (idx in AlreadymovieList):
            Movie_Ratgs.append((idx + 1, val))

    preds= sorted(Movie_Ratgs, key=last, reverse=True)[:min(count, len(Movie_Ratgs))]
    selected_ids=[]
    for p in preds:
        selected_ids.append(p[0])
    top_selected_movie_ids = selected_ids
    result = []
    for id in top_selected_movie_ids:
        # print(title_by_id[id])
        result.append(title_by_id[id])
    return result




preds = recommendSVD(45, 10)
print(preds)


def content_based(user):
    import pandas as pd
    import numpy as np
    from scipy import spatial

    debug = False
    flag = False
    movies_cols = []
    temp_genres = [i for i in range(5, 24)]
    movies_cols += temp_genres

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
        print(movies_by_user)
    # movie_title = 'Toy Story (1995)'
    # user = 1
    if user in df.user_id.values:
        movie_ids_selected = movies_by_user[user]
    else:
        movie_ids_selected = movies_by_user[1]
    # print movie_ids_selected
    ind_of_given_title = movie_ids_selected[0]
    print("Since the user has rated the movie '", title_by_id[ind_of_given_title], "' as 5 , so we recommend these movies.")
    print("-------------------CONTENT BASED ALGORITHM---------------------")
    print('User id: ', user)
    print("----------------------------------------------------------------")
    # if debug:
    #     print(ind_of_given_title)
    genre_of_inp_title = genres[ind_of_given_title]
    cosine_similarity_dict = {}
    for i in range(0, len(titles)):
        cosine_sim = 1 - spatial.distance.cosine(genre_of_inp_title, genres[i])
        cosine_similarity_dict[i] = cosine_sim
    if debug:
        print(cosine_similarity_dict)
    sorted_cos_sim_movie_id_pairs = sorted([(v, k) for (k, v) in cosine_similarity_dict.items()], reverse=True)
    top_selected = sorted_cos_sim_movie_id_pairs[:6]
    if debug:
        print(top_selected)
    top_selected_movie_ids = [top_selected[i][1] for i in range(len(top_selected))]
    if debug:
        print(top_selected_movie_ids)
    prstrg = "Movie Recommendations :".upper()
    print(prstrg)
    print()
    result = []
    for id in top_selected_movie_ids:
        if id != ind_of_given_title:
            print(title_by_id[id])
            result.append(title_by_id[id])
    return result


@app.route('/', methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        s = request.form.get('s')
        classifiers=False
        print(s)
        r = []

        cf_item = []
        cf_user = []
        knn=[]
        svd=[]
        mlp=[]
        cont_g_y=[]
        if (s).isdigit():
            print("True -------------")
            ids, cf_item = getRecommendation(int(s))
            ids, cf_user = getRecommended(int(s))
            r=content_based(int(s))
            cont_g_y=content_genre_year(int(s))
            svd = recommendSVD(int(s), 5)
            if classifiers:
                from mlp import mlp_recommender
                from KNN import knn_recommender
                knn = knn_recommender(int(s))

                mlp=mlp_recommender(int(s))
        # else:
        #     r = content_based(s)
        err = len(r)
        print("length :", len(r), len(cf_item), len(cf_user))

        if len(r) != 0 and len(cf_item) != 0 and len(cf_user) != 0:
            context = {
                'text': s,
                'cf_res': cf_item,
                'cf_user': cf_user,
                'res':r,
                # 'knn':knn,
                'svd':svd,
                # 'mlp':mlp,
                'cont':cont_g_y,

            }


        else:
            context = {
                'text': s,

                'error': 'Some error occurred!!!',
            }


        return render_template('iiitd/dmg/search.html', context=context)
    return render_template('iiitd/dmg/search.html')


if __name__ == '__main__':
    app.run()
