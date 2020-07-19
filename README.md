# anime-recommendation-engine

### abstract
Used data of animes and their ratings by individuals (source: MyAnimeList/Kaggle) and trained a recom-mendation engine on this data. Also made word cloud of genres for all the clusters found. Used KNNalgorithm and Python to perform this task.

### methodology
firstly i made groups of rating by user id, this was done to find the mean rating of every user as every users give ratings according to their personal preferences. then all the entries having entries less than that of mean rating of that user are removed as they are most likely to be not good animes for that user. 
then i merged user_id and ratings by anime_id to get information about amines(name). then we make the crosstab of anime name and user id marking each anime for every user 1 if he/she rated it else 0.
now as the shape of the data becomes very large to process(20000,7852) so i performed Principal Component Analysis on it to reshape it.
now at this point we have done all the required preprocessing on the data. now all that left was feeding to ML algorithm.
i used KNN algorithm to search for patterns and form cluster. i found 4 clusters in total. and then i gave each user its cluster number.
then for each cluster we find mean rating for every anime and sort them accordingly and extracts top 15 animes(by mean rating) and form word cloud of genre for every cluster.
