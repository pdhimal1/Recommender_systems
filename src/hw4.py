"""
Prakash Dhimal
Manav Garkel
George Mason University
CS 657 Mining Massive Datasets
Assignment 4: Recommender Systems

Use the MovieLens 20 M dataset
"""
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
from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField, FloatType
from sklearn.metrics.pairwise import cosine_similarity


def init_spark():
    conf = SparkConf()
    conf.set('spark.executor.memory', '15G')
    conf.set('spark.driver.memory', '15G')
    conf.set("spark.driver.host", "localhost")
    conf.setAppName("hw4")
    # conf.set('spark.driver.maxResultSize', '15G')

    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    return spark


def read_data(spark):
    # Read in the ratings csv
    ratings = spark.read.option("header", "true").csv('../data/ml-20m/ratings.csv')
    ratings = ratings.withColumn('userId', F.col('userId').cast(IntegerType()))
    ratings = ratings.withColumn('movieId', F.col('movieId').cast(IntegerType()))
    ratings = ratings.withColumn('rating', F.col('rating').cast(DoubleType()))
    ratings = ratings.select("userId", "movieId", "rating")
    return ratings


def create_cross_validator(rmse_evaluator, folds=3):
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    # coldStartStrategy="drop"
    als_model = ALS(itemCol='movieId',
                    userCol='userId',
                    ratingCol='rating',
                    nonnegative=True,
                    seed=5)

    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    # rank, maxIter
    ranks = [1, 2, 4, 8]  # number of features
    iterations = [10, 20]
    regParam = [0.0, 0.01, 0.3]  # 0.01 is the defualt
    param_grid = ParamGridBuilder() \
        .addGrid(als_model.rank, ranks) \
        .addGrid(als_model.maxIter, iterations) \
        .build()

    '''
        .addGrid(als_model.seed, seed) \
        .addGrid(als_model.regParam, regParam) \
        # this 
        .addGrid(als_model.implicitPrefs, implicitPrefs) \
    '''

    cross_validator = CrossValidator(estimator=als_model,
                                     estimatorParamMaps=param_grid,
                                     evaluator=rmse_evaluator,
                                     numFolds=folds)
    return cross_validator


def get_model(rmse_evaluator, rank=4, crossValidation=False, folds=3):
    if crossValidation:
        cross_validator = create_cross_validator(rmse_evaluator, folds=folds)
        return cross_validator
    else:
        return ALS(itemCol='movieId',
                   userCol='userId',
                   ratingCol='rating',
                   nonnegative=True,
                   rank=rank)


def als(rmse_evaluator, trainingDF, testDF, outFile, rank=4, crossValidation=False, folds=3):
    if crossValidation:
        print("Running ALS cross validation with {} folds ...".format(folds))
        print("ALS cross validation with {} folds ...".format(folds), file=outFile)
    else:
        print("Running ALS with rank={} ...".format(rank))
        print("ALS with rank={} ...".format(rank), file=outFile)

    time_start = time.time()
    model = get_model(rmse_evaluator, rank, crossValidation, folds=folds)
    model = model.fit(trainingDF)
    test_prediction = model.transform(testDF)
    time_end = time.time()

    if crossValidation:
        # train the ALS model for collaborative filtering
        # todo - Use the Products of Factors technique for your system and optimize the loss function with ALS.
        # See slides
        print("Best model selected from cross validation:\n", model.bestModel)
        print("Best model selected from cross validation:\n", model.bestModel, file=outFile)
        print("Folds: ", folds, file=outFile)
        print("took ", time_end - time_start, " seconds for cross validation")
        print("took ", time_end - time_start, " seconds for cross validation", file=outFile)
        print("took {} minutes for cross validation".format((time_end - time_start) / 60))
        print("took {} minutes for cross validation".format((time_end - time_start) / 60), file=outFile)
    else:
        print("took ", time_end - time_start, " seconds for ALS")
        print("took ", time_end - time_start, " seconds for ALS", file=outFile)
        print("took {} minutes for ALS".format((time_end - time_start) / 60))
        print("took {} minutes for ALS".format((time_end - time_start) / 60), file=outFile)

    print("ALS predictions are done!")

    test_prediction_with_na = test_prediction
    test_prediction = test_prediction.na.drop()

    # Generate top 10 movie recommendations for each user
    # userRecs = cross_validation_model.bestModel.recommendForAllUsers(5)
    # userRecs.show()
    # Generate top 10 user recommendations for each movie
    # movieRecs = cross_validation_model.bestModel.recommendForAllItems(5)
    # movieRecs.show()

    return test_prediction, test_prediction_with_na


def get_ratings(x, item_similarity, train_df, k):
    userID = x[0]
    movieID = x[1]
    # taking only those k users that have rated the movie
    this_item_distances = item_similarity[movieID]
    sorted_distances = this_item_distances.sort_values(ascending=False)[1:]
    # get the ratings by this user
    this_user = train_df[str(int(userID))]
    this_user.index = train_df.movieId

    ratings_this_user_this_movie = []
    for key in sorted_distances.keys():
        if len(ratings_this_user_this_movie) >= k:
            break
        this_user_this_movie = this_user[key]
        if this_user_this_movie > 0:
            ratings_this_user_this_movie.append(this_user_this_movie)

    item_rating = mean(ratings_this_user_this_movie)
    return np.float16(item_rating)


def item_item_collaborative_filtering(k, ratings, testDF):
    ratings.cache()
    index = ratings.toPandas().movieId.unique()
    pivoted = ratings.groupBy("movieId").pivot('userId').sum('rating').na.fill(0)
    pivoted.cache()
    pivoted_df = pivoted.toPandas()
    print("Matrix creation done ...")
    item_similarity = cosine_similarity(pivoted_df)
    item_similarity = pd.DataFrame(item_similarity)
    item_similarity.index = index
    item_similarity.columns = index
    print("Item Item similarity matrix creation done ...")

    test_data = testDF.rdd.map(tuple)
    item_item_results = test_data.map(
        lambda x: (x[0], x[1], x[2], float(get_ratings(x, item_similarity, pivoted_df, k))))
    return item_item_results

    # testDF.cache()
    # test_data = testDF.toPandas()
    # item_item_collaborative_labels = [get_ratings(x, item_similarity, pivoted_df, k) for x in test_data[:].iterrows()]
    # test_data['prediction_item_item_cf'] = item_item_collaborative_labels
    # return test_data


def main(data_size, k, outFile, time_stamp, cf=False, rank=4, crossValidation=False, folds=3):
    spark = init_spark()
    ratings = read_data(spark)

    # Let's split our data into training data and testing data
    ratings = ratings.limit(data_size)  # total dataset is 20000263
    print("Data size: ", ratings.count())
    print("Data size: ", ratings.count(), file=outFile)
    trainTest = ratings.randomSplit([0.8, 0.2])

    trainingDF = trainTest[0]
    testDF = trainTest[1]
    time_start = time.time()
    # Evaluate model, can I give it two metrics?
    rmse_evaluator = RegressionEvaluator(
        predictionCol="prediction",
        labelCol="rating",
        metricName="rmse")
    mae_evaluator = RegressionEvaluator(
        predictionCol="prediction",
        labelCol="rating",
        metricName="mae")
    mse_evaluator = RegressionEvaluator(
        predictionCol="prediction",
        labelCol="rating",
        metricName="mse")

    test_prediction, test_prediction_with_na = als(
        rmse_evaluator,
        trainingDF,
        testDF,
        outFile,
        rank,
        crossValidation,
        folds=folds)
    test_prediction.show()
    print("ALS RMSE: ", rmse_evaluator.evaluate(test_prediction))
    print("ALS RMSE: ", rmse_evaluator.evaluate(test_prediction), file=outFile)
    print("ALS MAE: ", mae_evaluator.evaluate(test_prediction))
    print("ALS MAE: ", mae_evaluator.evaluate(test_prediction), file=outFile)
    print("ALS MSE: ", mse_evaluator.evaluate(test_prediction))
    print("ALS MSE: ", mse_evaluator.evaluate(test_prediction), file=outFile)

    if cf:
        print("Running item-item collaborative filtering ...")
        time_start_cf = time.time()
        prediction_item_item = item_item_collaborative_filtering(k, ratings, testDF)
        time_end = time.time()
        print("took {} minutes for item-item collaborative filtering.".format((time_end - time_start_cf) / 60))
        print("took {} minutes for item-item collaborative filtering.".format((time_end - time_start_cf) / 60),
              file=outFile)
        schema = StructType([StructField('userId', IntegerType(), True),
                             StructField('movieId', IntegerType(), True),
                             StructField('rating', FloatType(), True),
                             StructField('prediction', FloatType(), True)])
        prediction_item_item_df = spark.createDataFrame(prediction_item_item, schema)

        print("Item-Item CF RMSE: ", rmse_evaluator.evaluate(prediction_item_item_df))
        print("Item-Item CF RMSE: ", rmse_evaluator.evaluate(prediction_item_item_df), file=outFile)
        print("Item-Item CF MAE: ", mae_evaluator.evaluate(prediction_item_item_df))
        print("Item-Item CF MAE: ", mae_evaluator.evaluate(prediction_item_item_df), file=outFile)
        print("Item-Item CF MSE: ", mse_evaluator.evaluate(prediction_item_item_df))
        print("Item-Item CF MSE: ", mse_evaluator.evaluate(prediction_item_item_df), file=outFile)

        test_prediction_with_na = test_prediction_with_na.withColumnRenamed("prediction", "prediction_als")
        prediction_item_item_df = prediction_item_item_df.withColumnRenamed("prediction", "prediction_item_item_cf")
        prediction_item_item_df.cache()
        prediction_item_item_df_pandas = prediction_item_item_df.toPandas()

        prediction_total = prediction_item_item_df_pandas.merge(
            test_prediction_with_na.toPandas(),
            on=['userId', 'movieId', 'rating'])
        print(prediction_total)

        # should probably give 60% to item item
        prediction_total['prediction'] = prediction_total[['prediction_item_item_cf', 'prediction_als']].mean(axis=1)
        print(prediction_total)

        prediction_total_df = spark.createDataFrame(prediction_total)
        print("Hybrid RMSE: ", rmse_evaluator.evaluate(prediction_total_df))
        print("Hybrid RMSE: ", rmse_evaluator.evaluate(prediction_total_df), file=outFile)
        print("Hybrid MAE: ", mae_evaluator.evaluate(prediction_total_df))
        print("Hybrid MAE: ", mae_evaluator.evaluate(prediction_total_df), file=outFile)
        print("Hybrid MSE: ", mse_evaluator.evaluate(prediction_total_df))
        print("Hybrid MSE: ", mse_evaluator.evaluate(prediction_total_df), file=outFile)

        file_name = '../out/predictions_total-' + time_stamp + "-" + str(data_size) + '.csv'
        prediction_total.to_csv(file_name, index=False)

    print("Total time to run Script: {} minutes".format((time.time() - time_start) / 60))
    print("Total time to run Script: {} minutes".format((time.time() - time_start) / 60), file=outFile)
    spark.stop()


if __name__ == "__main__":
    # todo parameters
    data_size = 1000  # 1000000 # 10000000  # total dataset is 20000263
    # for als if not cross validation
    rank = 1
    # cross validation
    crossValidation = False
    folds = 5
    # item item collaborative filtering
    cf = True
    k = 15

    time_stamp = str(int(time.time()))
    out_file_name = '../out/output-' + time_stamp + "-" + str(data_size) + '.txt'
    out_file = open(out_file_name, 'w')
    main(data_size,
         k,
         out_file,
         time_stamp,
         cf=cf,
         rank=rank,
         crossValidation=crossValidation,
         folds=folds)
    print("Check ", out_file.name)
