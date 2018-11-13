# include utf-coding8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Function that get movie recommendations based on the cosine similarity score of movie genres


def main():

    import json
    with open("genres.json") as fp:
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
    title = 'Toy Story (1995)'
    print('Input: ' + title)
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]

    movie_recommendations = titles.iloc[movie_indices]
    print(movie_recommendations)


if __name__ == '__main__':
    main()
