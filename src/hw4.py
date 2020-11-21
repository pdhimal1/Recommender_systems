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
from pyspark.sql.types import DoubleType, IntegerType, StructType, StructField, FloatType
from sklearn.metrics.pairwise import cosine_similarity

conf = SparkConf()
conf.set('spark.executor.memory', '15G')
conf.set('spark.driver.memory', '15G')
conf.set("spark.driver.host", "localhost")
conf.setAppName("hw4")
# conf.set('spark.driver.maxResultSize', '15G')

sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# Read in the ratings csv
ratings = spark.read.option("header", "true").csv('../data/ml-20m/ratings.csv')
ratings = ratings.withColumn('userId', F.col('userId').cast(IntegerType()))
ratings = ratings.withColumn('movieId', F.col('movieId').cast(IntegerType()))
ratings = ratings.withColumn('rating', F.col('rating').cast(DoubleType()))

ratings = ratings.select("userId", "movieId", "rating")

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
    .build()

'''
    .addGrid(als_model.seed, seed) \
    .addGrid(als_model.regParam, regParam) \
    # this 
    .addGrid(als_model.implicitPrefs, implicitPrefs) \
'''
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
# ratings = ratings.limit(1000)  # total dataset is 20000263
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
print("took ", (time_end - time_start)/60, " minutes for cross validation")
test_prediction_with_na = test_prediction
test_prediction = test_prediction.na.drop()

test_prediction.show()
print("ALS RMSE: ", rmse_evaluator.evaluate(test_prediction))

mae_evaluator = RegressionEvaluator(
    predictionCol="prediction",
    labelCol="rating",
    metricName="mae")
print("ALS MAE: ", mae_evaluator.evaluate(test_prediction))

mse_evaluator = RegressionEvaluator(
    predictionCol="prediction",
    labelCol="rating",
    metricName="mse")
print("ALS MSE: ", mse_evaluator.evaluate(test_prediction))

# # Generate top 10 movie recommendations for each user
# userRecs = cross_validation_model.bestModel.recommendForAllUsers(5)
# userRecs.show()
# # Generate top 10 user recommendations for each movie
# movieRecs = cross_validation_model.bestModel.recommendForAllItems(5)
# movieRecs.show()

def fill_ratings(this_user, movieID, col):
    rating = this_user[this_user.movieId == movieID].rating.iloc[0]
    if math.isnan(rating):
        rating = 0
    col[movieID] = rating

def fill_Utility_matrix(data, unique_movies, user, utility_matrix):
    # np array of len()
    col = dict.fromkeys(unique_movies, 0)
    this_user = data[data.userId == user]
    if not this_user.empty:
        [fill_ratings(this_user, movieID, col) for movieID in this_user.movieId]
    utility_matrix[user] = col

def get_Matrix(data):
    unique_users = data.userId.unique()
    unique_movies = data.movieId.unique()
    print("Unique users: ", len(unique_users))
    print("Unique movies: ", len(unique_movies))
    utility_matrix = {}
    time_start = time.time()
    [fill_Utility_matrix(data, unique_movies, user, utility_matrix) for user in unique_users]
    time_end = time.time()
    print("took {} minutes for creating the matrix.".format((time_end - time_start) / 60))
    return pd.DataFrame(utility_matrix)


# Item - Item CF predictions
def item_item_cf(x, item_similarity, train_df):
    userID = x[0]
    movieID = x[1]

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

    return item_rating


# Spark Implementation for Item-Item CF
train_df = get_Matrix(ratings.toPandas())
item_item_index = train_df.index
item_similarity = cosine_similarity(train_df)
item_similarity = pd.DataFrame(item_similarity)
item_similarity.index = item_item_index
item_similarity.columns = item_item_index

k = 10
test_data = testDF.rdd.map(tuple)
item_item_results = test_data.map(lambda x: (x[0], x[1], x[2], float(item_item_cf(x, item_similarity, train_df))))
schema = StructType([StructField('userId', IntegerType(), True),
                     StructField('movieId', IntegerType(), True),
                     StructField('rating', FloatType(), True),
                     StructField('prediction_item_item_cf', FloatType(), True)])
prediction_item_item = spark.createDataFrame(item_item_results, schema)


# Print Stats for Item-Item CF
prediction_item_item_df = prediction_item_item.withColumnRenamed("prediction_item_item_cf", "prediction")
print("Item-Item CF RMSE: ", rmse_evaluator.evaluate(prediction_item_item_df))
print("Item-Item CF MAE: ", mae_evaluator.evaluate(prediction_item_item_df))
print("Item-Item CF MSE: ", mse_evaluator.evaluate(prediction_item_item_df))


# Hybrid Approach
test_prediction_with_na = test_prediction_with_na.withColumnRenamed("prediction", "prediction_als")
prediction_total = prediction_item_item.join(test_prediction_with_na, on=['userId', 'movieId', 'rating'])
prediction_total.show()

# todo - can define a custom function to handle NAN's. Currently just chose to drop them.
prediction_total = prediction_total.withColumn("prediction",
                                               (prediction_total.prediction_item_item_cf + prediction_total.prediction_als)/2).dropna()
prediction_total.show()

# Print Stats for Hybrid Approach
print("Hybrid RMSE: ", rmse_evaluator.evaluate(prediction_total))
print("Hybrid MAE: ", mae_evaluator.evaluate(prediction_total))
print("Hybrid MSE: ", mse_evaluator.evaluate(prediction_total))

print("Total Time to run Script {} minutes".format((time.time() - time_start)/60))

sc.stop()
