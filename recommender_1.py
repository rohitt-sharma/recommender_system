
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



# --------------------------------Top Treding recommender-------------------


# Weighted average (wr) = (v/(v+m).R) + (m/(v+m).C)   
# m = min votes required to be on list
# v = number of votes for movie
# R = average rating of the movie
# C = mean vote across the whole report

# Finding mean rating
c = df2["vote_average"].mean()

print("Average rating",c)


# deciding miimum number of cotes "m"

m = df2["vote_count"].quantile(0.9)

print("Vote count",m)


# Filtering the movies for the chart
q_movies = df2.copy().loc[df2["vote_count"]>=m]

print(q_movies.shape)


def weighted_rating(x,m=m, c=c):
    V = x["vote_count"]
    R = x["vote_average"]

    # calculation based on the formula
    return (V/(V+m)*R) + (m/(m+V)*c)


# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)  # weighted_rating function applied to each row (axis = 1)

#Sort movies based on score calculated above
#q_movies = q_movies.sort_values('score', ascending=False)  # sorting the values in ascending manner

#Print the top 10 movies based on ratings
#print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10))   # top 10 movies are shown here


#Print the top 10 movies based on famous
q_movies = q_movies.sort_values('popularity', ascending=False)  

print(q_movies[['title', 'vote_count', 'vote_average', 'score','popularity']].head(10))  # top 10 movies are shown here

