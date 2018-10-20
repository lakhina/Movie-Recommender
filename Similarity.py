# Please change file pathname for u.data and u.item before running the demo
import pandas as panda

r_cols = ['user_id', 'movie_id', 'rating']
# Change path to run
ratings = panda.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
                         usecols=range(3))

m_cols = ['movie_id', 'title']
# Change path to run
movies = panda.read_csv('ml-100k/u.item', sep='|', names=m_cols,
                        usecols=range(2))

ratings = panda.merge(movies, ratings)

# print ratings.head(10)
# ratings.tail(10)

userRatings = ratings.pivot_table(index=['user_id'], columns=['title'], values='rating')
# print userRatings.head()

corrMatrix = userRatings.corr()
# print corrMatrix.head()

corrMatrix = userRatings.corr(method='pearson', min_periods=50)
print "Corr matrix"
print corrMatrix.head()
"""User ratings for second user"""
myRatings = userRatings.loc[2].dropna()
print "my ratings"
print myRatings
print (myRatings).shape
#
simCandidates = panda.Series()
for i in range(0, len(myRatings.index)):
    # print "adding sims for " + myRatings.index[i] + "..."
    sims = corrMatrix[myRatings.index[i]].dropna()
    sims = sims.map(lambda x: x * myRatings[i])
    simCandidates = simCandidates.append(sims)

print "sorting in decreasing order of similarity score: "
simCandidates.sort_values(inplace=True, ascending=False)
print "Similarity candidates"
print simCandidates.head(10)

simCandidates = simCandidates.groupby(simCandidates.index).sum()

simCandidates.sort_values(inplace=True, ascending=False)
print "\n"
print "---Adding up the similarity scores of duplicate values:---"
print simCandidates.head(10),

filteredSims = simCandidates.drop(myRatings.index, errors='ignore')
print "\n"
print "---Filtering the result to remove already rated movies:---"
print filteredSims.head()
print len(filteredSims)
