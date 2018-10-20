# include utf-coding8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Function that get movie recommendations based on the cosine similarity score of movie genres
def genre_recommendations(title, indices, cosine_sim, titles):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


def main():
    # Reading ratings file
    # Ignore the timestamp column
    # ratings = pd.read_csv('data/ratings.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating'])

    # Reading users file
    # users = pd.read_csv('data/users.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

    # Reading movies file
    # movies = pd.read_csv('data/movies_genres.csv', encoding='latin-1')
    # print movies.head()
    # tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    # tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0)
    # tfidf_matrix = tf.fit_transform(movies[1])
    #
    # cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    # from scipy import spatial
    #
    # # Build a 1-dimensional array with movie titles
    # titles = movies['title']
    # indices = pd.Series(movies.index, index=movies['title'])
    # title = 'Good Will Hunting (1997)'
    # print('Input: ' + title)
    # movie_recommendations = genre_recommendations(title, indices, cosine_sim, titles).head(20)
    # print(movie_recommendations)


    import json
    import numpy
    with open("genres.json") as fp:
        genres = json.load(fp)
    print numpy.array(genres)[:10]
    import pandas as pd
    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, usecols=[1])
    print movies.head()
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0)
    tfidf_matrix = tf.fit_transform(genres)

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    titles = movies[1]
    indices = pd.Series(movies.index, index=movies[1])
    title = 'Good Will Hunting (1997)'
    print('Input: ' + title)
    movie_recommendations = genre_recommendations(title, indices, cosine_sim, titles).head(20)
    print(movie_recommendations)


if __name__ == '__main__':
    main()
