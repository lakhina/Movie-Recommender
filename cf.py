import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics.pairwise import pairwise_distances 




#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols)
print (ratings.user_id.unique())
print (ratings.user_id.unique().shape)
number_of_users = ratings.user_id.unique().shape[0]
number_of_items = ratings.movie_id.unique().shape[0]


#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols) #,encoding='latin-1')


ratings_train = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols)
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols)
print (ratings_train.shape, ratings_test.shape)



# Forming main data matrix user item matrix
data_matrix = np.zeros((number_of_users, number_of_items))
for line in ratings.itertuples():
	#print (line)
	data_matrix[line[1]-1, line[2]-1] = line[3]                  # first index in pandas dataframe is unique id, hence started from 1, storing the rating given by user to a particular item
    


print (data_matrix.shape)  
#print (data_matrix)   

user_similarity = 1-pairwise_distances(data_matrix, metric='correlation')
item_similarity = 1-pairwise_distances(data_matrix.T, metric='correlation')

print (user_similarity.shape)
#print (user_similarity)









def getRecommended(user):                                                         # using user user similarity
	recommended=set() 
	tempRecommended=set()
	top_5_idx = np.argsort(user_similarity[user])[-6:].tolist()                        # getting top 6 similar users, will remove top that is the element itself, and 5 will be returned
	top_5_values = [user_similarity[user][i] for i in top_5_idx]
	top_5_idx.reverse()
	top_5_idx.pop(0)
	print (top_5_values)
# add already rated movies by the user too
	i=0
	#user=1
	for val in data_matrix[user]:
		i+=1
		if (val!=0):
			tempRecommended.add(i)
			print (val)	
	print ("recommended originl----- ",tempRecommended)
	#tempRecommended=recommended.copy()
	count=0
	rating5=set()
	rating4=set()
	rating3=set()
	rating2=set()
	rating1=set()
	for idx in top_5_idx:
		i=-1
		temp= data_matrix[idx]
		for val in temp:
			i+=1
			if (val==5):
				print ("here 5 ",i)
				if (i not in tempRecommended):
					print ("hello")
					count+=1
					recommended.add(i)
				rating5.add(i)
			
			if (val==4):
				rating4.add(i)
			
			if (val==3):
				rating3.add(i)
			
			if (val==2):
				rating2.add(i)
			
			if (val==1):
				rating1.add(i)
		
			if (count>=5):
				break
		if (count>=5):
			break
		
		
	while (count<5):
		j=-1
		for v in rating4:
			j+=1
			if (count>=5):
				break
			if (j not in tempRecommended):
				recommended.add(j)
				count+=1
		j=-1
		for v in rating3:
			j+=1
			if (count>=5):
				break
			if (j not in tempRecommended):
				recommended.add(j)
				count+=1
		j=-1
		for v in rating2:
			j+=1
			if (count>=5):
				break
			if (j not in tempRecommended):
				recommended.add(j)
				count+=1
		j=-1
		for v in rating1:
			j+=1
			if (count>=5):
				break
			if (j not in tempRecommended):
				recommended.add(j)
				count+=1
		if (count>=5):
			break
	return recommended


#getRecommended(134)		

print (" ****************************************************************************************************** ")
#print (recommended)




def getRecommendation(user):                                # using item item similarity
	recommended=set() 
	tempRecommended=set()
	rating5=set()
	rating4=set()
	rating3=set()
	rating2=set()
	rating1=set()
	i=-1
	count=0
	temp= data_matrix[user]
	for val in temp:
			i+=1
			if (val==5):
				print ("here 5 ",i)
				#if (i not in tempRecommended):
					#print ("hello")
				count+=1
				recommended.add(i)
				rating5.add(i)
			
			if (val==4):
				rating4.add(i)
			
			if (val==3):
				rating3.add(i)
			
			if (val==2):
				rating2.add(i)
			
			if (val==1):
				rating1.add(i)
		
			if (count>=5):
				break
	
		
		
	while (count<5):
		j=-1
		for v in rating4:
			j+=1
			if (count>=5):
				break
			
			recommended.add(j)
			count+=1
		j=-1
		for v in rating3:
			j+=1
			if (count>=5):
				break
			
			recommended.add(j)
			count+=1
		j=-1
		for v in rating2:
			j+=1
			if (count>=5):
				break
			
			recommended.add(j)
			count+=1
		j=-1
		for v in rating1:
			j+=1
			if (count>=5):
				break
			
			recommended.add(j)
			count+=1
		
		if (count>=5):
			break
	print (recommended)
	final_recommended=set()
	for item in recommended:
		top_5_idx = np.argsort(item_similarity[item])[-6:].tolist()                   # getting top 6 similar users, will remove top that is the element itself, and 5 will be returned
		print ("------------------- ",top_5_idx)
		top_5_values = [item_similarity[item][i] for i in top_5_idx]
		print (top_5_values)
		top_5_idx.reverse()
		top_5_idx.pop(0)
		el=top_5_idx.pop(0)
		while (el in final_recommended and len(top_5_idx)!=0):
			el=top_5_idx.pop(0)
		final_recommended.add(el)
	print ("........................................ ",final_recommended)
	return final_recommended	
		
	

print (" item ")
print (" @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@22 ",getRecommendation(4))

print (" user ")
print (" ##################################################################################################3 ",getRecommended(4))
			
'''

print (top_5_idx)
lst=data_matrix[top_5_idx[3]]
for i in range (0,len(lst)):
	if (lst[i]!=0):
		print (lst[i])
	
'''

