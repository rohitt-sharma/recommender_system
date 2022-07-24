

import pandas as pd
import numpy as np

df1 = pd.read_csv("datas/tmdb_5000_credits.csv")
df2 = pd.read_csv("datas/tmdb_5000_movies.csv")

df1.columns = ["id","title","cast","crew"]  # for renaming the columns to as mentioned in the command, the sequence is also as written

#print(df2)
#print("-x-x-x-x-x-x-x-")

df2 = df2.merge(df1,on="title")

#print(df2.head(5))

print(df2.columns.values)


#################################################################################

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df2['overview'] = df2['overview'].fillna('')


#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['overview'])

#Output the shape of tfidf_matrix
print(tfidf_matrix.shape)

#######-x-x-x-x-x-x-x-x-x-x-x-x-x-x--x-x-x-x-x-x-x--x-x-x-x-x-x-x--x-x-x-x-x-x-x-x-x-x-x-x-
# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


#Construct a reverse map of indices and movie titles
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()



# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return print(df2['title'].iloc[movie_indices])



get_recommendations('The Avengers')




