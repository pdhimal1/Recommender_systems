"""
Prakash Dhimal
George Mason University
CS 657 Mining Massive Datasets
Assignment 4: Recommender Systems

Use the MovieLens 20 M dataset
"""
import math
import time
from statistics import mean

import numpy as np
import pandas as pd
from pyspark import SparkContext, SparkConf
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType
from sklearn.metrics.pairwise import cosine_similarity

conf = SparkConf()
conf.setMaster('local[*]')
conf.set('spark.executor.memory', '15G')
conf.set('spark.driver.memory', '15G')
conf.setAppName("hw4")
# conf.set('spark.driver.maxResultSize', '15G')

sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# Read in the ratings csv
ratings = spark.read.option("header", "true").csv('./data/ml-20m/ratings.csv')
ratings = ratings.withColumn('userId', F.col('userId').cast(IntegerType()))
ratings = ratings.withColumn('movieId', F.col('movieId').cast(IntegerType()))
ratings = ratings.withColumn('rating', F.col('rating').cast(DoubleType()))

ratings = ratings.select("userId", "movieId", "rating")
ratings.show()

print("Running ALS cross validation ...")
# train the ALS model for collaborative filtering
# todo - Use the Products of Factors technique for your system and optimize the loss function with ALS.
# See slides
'''
If the rating matrix is derived from another source of information (i.e. it is inferred from other signals),
you can set implicitPrefs to True to get better results:

regParam=0.01
'''
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
# coldStartStrategy="drop"
als_model = ALS(itemCol='movieId',
                userCol='userId',
                ratingCol='rating',
                nonnegative=True)

# We use a ParamGridBuilder to construct a grid of parameters to search over.
# todo - add other parameters here?
# rank, maxIter
seed = [5]  # Random seed for initial matrix factorization model
ranks = [4, 8, 12]  # number of features
iterations = [10, 20]
regParam = [0.01]  # 0.01 is the defualt
implicitPrefs = [True, False]
coldStartStrategy = ["drop", "nan"]
param_grid = ParamGridBuilder() \
    .addGrid(als_model.rank, ranks) \
    .addGrid(als_model.maxIter, iterations) \
    .addGrid(als_model.seed, seed) \
    .addGrid(als_model.regParam, regParam) \
    .addGrid(als_model.implicitPrefs, implicitPrefs) \
    .build()

# Evaluate model, can I give it two metrics?
rmse_evaluator = RegressionEvaluator(
    predictionCol="prediction",
    labelCol="rating",
    metricName="rmse")

cross_validator = CrossValidator(estimator=als_model,
                                 estimatorParamMaps=param_grid,
                                 evaluator=rmse_evaluator,
                                 numFolds=3)  # todo - change the number of folds

# Let's split our data into training data and testing data
# todo - control the amount of data here
print("Total dataset: ", ratings.count())
ratings = ratings.limit(1000)  # total dataset is 20000263
trainTest = ratings.randomSplit([0.8, 0.2])

trainingDF = trainTest[0]
testDF = trainTest[1]

time_start = time.time()
# Run cross-validation, and choose the best set of parameters.
cross_validation_model = cross_validator.fit(trainingDF)

# Make predictions on test documents. cvModel uses the best model found (lrModel).
test_prediction = cross_validation_model.transform(testDF)
# test_prediction.cache()
time_end = time.time()
print("ALS predictions are done!")
print("Best model selected from cross validation:\n", cross_validation_model.bestModel)
print("took ", time_end - time_start, " seconds for cross validation")
test_prediction_with_na = test_prediction
test_prediction = test_prediction.na.drop()

test_prediction.show()

score = rmse_evaluator.evaluate(test_prediction)
print("RMSE: ", score)

mae_evaluator = RegressionEvaluator(
    predictionCol="prediction",
    labelCol="rating",
    metricName="mae")
print("MAE: ", mae_evaluator.evaluate(test_prediction))

mse_evaluator = RegressionEvaluator(
    predictionCol="prediction",
    labelCol="rating",
    metricName="mse")
print("MSE: ", mse_evaluator.evaluate(test_prediction))

# Generate top 10 movie recommendations for each user
userRecs = cross_validation_model.bestModel.recommendForAllUsers(5)
userRecs.show()
# Generate top 10 user recommendations for each movie
movieRecs = cross_validation_model.bestModel.recommendForAllItems(5)
movieRecs.show()


def get_Matrix(data):
    unique_users = data.userId.unique()
    unique_movies = data.movieId.unique()
    utility_matrix = {}
    for user in unique_users:
        # np array of len()
        col = dict.fromkeys(unique_movies, 0)
        this_user = data[data.userId == user]
        if not this_user.empty:
            for movieID in this_user.movieId:
                # instead of this put the actual ratings in
                rating = this_user[this_user.movieId == movieID].rating.iloc[0]
                if math.isnan(rating):
                    rating = 0
                col[movieID] = rating
        utility_matrix[user] = col

    return pd.DataFrame(utility_matrix)


def item_item_collaborative_filtering():
    k = 25
    train_df = get_Matrix(ratings.toPandas())
    item_item_index = train_df.index
    item_similarity = cosine_similarity(train_df)
    item_similarity = pd.DataFrame(item_similarity)
    item_similarity.index = item_item_index
    item_similarity.columns = item_item_index

    test_data = testDF.toPandas()
    item_item_collaborative_labels = []
    for x in test_data[:].iterrows():
        userID = x[1]['userId']
        movieID = x[1]['movieId']
        # taking only those k users that have rated the movie
        this_item_distances = item_similarity[movieID]
        sorted_distances = this_item_distances.sort_values(ascending=False)[1:]
        # get the ratings by this user
        this_user = train_df[userID]

        ratings_this_user_this_movie = []
        for key in sorted_distances.keys():
            if len(ratings_this_user_this_movie) >= k:
                break
            this_user_this_movie = this_user[key]
            if this_user_this_movie > 0:
                ratings_this_user_this_movie.append(this_user_this_movie)

        item_rating = mean(ratings_this_user_this_movie)
        item_item_collaborative_labels.append(np.float16(item_rating))
    test_data['prediction_item_item_cf'] = item_item_collaborative_labels
    return test_data


prediction_item_item = item_item_collaborative_filtering()
prediction_total = prediction_item_item.merge(test_prediction_with_na.toPandas(), on=['userId', 'movieId', 'rating'])
print(prediction_total)

prediction_total['avg_prediction'] = prediction_total[['prediction-item-item-cf', 'prediction']].mean(axis=1)
print(prediction_total)

sc.stop()
# recommendations for a given user
'''
Item-Item collaborative filtering
The idea here is to find a set of movies similar to a given movie, 
and rate the given movie based on how those similar movies have been rated by the user.

user = 25
testDF.show()
userDf = testDF.filter(testDF.userId == user)
userDf.show(10)
mov = trainingDF.select('movieId', 'userId').subtract(userDf.select('movieId'))
mov.show(20)

# Again we need to covert our dataframe into RDD
pred_rat = cross_validation_model.transform(mov).collect()
# Get the top recommendations
recommendations = sorted(pred_rat, key=lambda x: x[2], reverse=True)[:5]
print(recommendations)

Using Spark, design and implement a Recommender System to predict ratings of movies. 
Base your utility matrix on the Movie Lens 20M dataset. 
Use the Products of Factors technique for your system and optimize the loss function with ALS .

Done?
To tune your solution, use crossvalidation over a subset (80%) of the data.
You can use the utilities in Spark to generate the folds. 
Use the rest of the dataset (20%) to test the system after tuning.Compute RMSE, MSE, and MAP.

TODO -
Implement a hybrid system that uses the ALS solution and item-item CF. 
You can use as guidance the scripts Hybrid Alg. and Hybrid Testing changing statements where appropriate.

item-item CF
Item item collaborative filtering

Item-Item collaborative filtering
The idea here is to find a set of movies similar to a given movie, 
and rate the given movie based on how those similar movies have been rated by the user.

For a given movie B:
Find a set of other movies that are similar to this movie. 
We achieve this by deploying the same techniques used in K-nearest neighbors.
Estimate this user’s (A’s) ratings based how it rated K nearest neighbors of the movie B.

Finding similar users:
#todo use some other sim matrix (Jaccord?) 
# cosine is probably better anyways
# centered cosine similarity? also known as pearson corelaton
I used Cosine similarity on the Utility matrix to find movies similar to a given movie. 
I precomputed the cosine similarity beforehand.

Predicting the ratings:
From the ratings obtained from the K nearest neighbors, we can take the average of ratings from the k nearest neighbors.

'''
