"""
Prakash Dhimal
Manav Garkel
George Mason University
CS 657 Mining Massive Datasets
Assignment 4: Recommender Systems

Use the MovieLens 20 M dataset
"""
import math
import time
from statistics import mean, StatisticsError

import numpy as np
import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType
from sklearn.metrics.pairwise import cosine_similarity


def init_spark():
    spark = SparkSession.builder.master("local[*]").appName("hw4").config("spark.driver.memory", "32g").getOrCreate()
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
    test_prediction.cache()
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


def get_ratings(userID, movieID, item_similarity, train_df, k):
    # taking only those k users that have rated the movie
    this_item_distances = item_similarity[movieID]
    sorted_distances = this_item_distances.sort_values(ascending=False)[1:]
    # get the ratings by this user
    this_user = train_df[int(userID)]
    # this_user.index = train_df.movieId

    ratings_this_user_this_movie = []
    for key in sorted_distances.keys():
        if len(ratings_this_user_this_movie) >= k:
            break
        this_user_this_movie = this_user[key]
        if this_user_this_movie > 0:
            ratings_this_user_this_movie.append(this_user_this_movie)

    item_rating = mean(ratings_this_user_this_movie)
    return float(item_rating)


def item_item_collaborative_filtering(k, dataSize, testDF):
    # get unique values in a column
    ratings = pd.read_csv('../data/ml-20m/ratings.csv')
    ratings = ratings.drop("timestamp", axis=1)
    ratings = ratings[:dataSize]
    pivoted = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    print("Matrix creation done ...")

    item_similarity = cosine_similarity(pivoted)
    item_similarity = pd.DataFrame(item_similarity)
    item_similarity.index = pivoted.index
    item_similarity.columns = pivoted.index
    print("Item Item similarity matrix creation done ...")

    udf_test_function = F.udf(lambda x, y: get_ratings(
        x,
        y,
        item_similarity,
        pivoted,
        k), DoubleType())
    item_item_results_df = testDF.withColumn("prediction", udf_test_function("userId", "movieId"))
    return item_item_results_df


def get_ratings_pyspark1(userID, movieId, pivoted, cosine_similarities_matrix, column_to_sims_dict, sims_to_column_dict,
                        k):
    this_movie = cosine_similarities_matrix.filter(
        cosine_similarities_matrix.movie1 == column_to_sims_dict[str(int(movieId))])
    cosine_similarities_movies = this_movie.sort(this_movie.sim.desc()).select('movie2').limit(k * 10).toPandas()

    this_user = pivoted.select(['movieId', str(int(userID))]).toPandas().set_index('movieId')

    ratings_this_user_this_movie = []
    for x in cosine_similarities_movies.iterrows():
        if len(ratings_this_user_this_movie) >= k:
            break
        key_movie = x[1]['movie2']
        actual_key = sims_to_column_dict[int(key_movie)]
        this_user_this_movie = this_user.loc[int(actual_key)][0]
        if this_user_this_movie > 0:
            ratings_this_user_this_movie.append(this_user_this_movie)

    try:
        item_rating = mean(ratings_this_user_this_movie)
    except StatisticsError:
        item_rating = 0
    print("MovieID: ", movieId, " UserID ", userID, " Rating: ", item_rating)
    return np.float16(item_rating)


def item_item_collaborative_filtering_pyspark1(k, ratings, testDF):
    pivoted = ratings.groupBy("movieId").pivot('userId').sum('rating').na.fill(0)
    pivoted_T = ratings.groupBy("userId").pivot('movieId').sum('rating').na.fill(0)

    pivoted_dropped = pivoted_T.drop(pivoted_T.userId)
    columns = pivoted_dropped.columns
    print("Creating similarity dataframe ...")
    column_to_sims_dict = dict(zip(columns, range(len(columns))))
    sims_to_column_dict = dict(zip(range(len(columns)), columns))
    mat = RowMatrix(pivoted_dropped.rdd.map(tuple))
    simsPerfect = mat.columnSimilarities()
    similarity_df = simsPerfect.entries.toDF()
    similarity_df = similarity_df.toDF(*['movie1', 'movie2', 'sim'])
    print("Similarity matrix creation done ...")

    udf_test_function = F.udf(lambda x, y: get_ratings_pyspark1(
        x,
        y,
        pivoted,
        similarity_df,
        column_to_sims_dict,
        sims_to_column_dict,
        k), DoubleType())
    prediction_total_df = testDF.withColumn("prediction", udf_test_function("userId", "movieId"))
    prediction_total_df.show()

    return prediction_total_df


def get_ratings_pyspark(x, pivoted, cosine_similarities_matrix, columns, k):
    userID = x[1]['userId']
    movieID = x[1]['movieId']
    this_movie = cosine_similarities_matrix.filter(
        cosine_similarities_matrix.movie1 == columns.index(str(int(movieID))))
    cosine_similarities_movies = this_movie.sort(this_movie.sim.desc()).select('movie2').limit(k * 100).toPandas()

    this_user = pivoted.select(['movieId', str(int(userID))]).toPandas().set_index('movieId')

    ratings_this_user_this_movie = []
    for x in cosine_similarities_movies.iterrows():
        key_movie = x[1]['movie2']
        actual_key = columns[key_movie]
        this_user_this_movie = this_user.loc[int(actual_key)][0]
        if this_user_this_movie > 0:
            ratings_this_user_this_movie.append(this_user_this_movie)

    try:
        item_rating = mean(ratings_this_user_this_movie)
    except StatisticsError:
        item_rating = 0
    print("MovieID: ", movieID, " UserID ", userID, " Rating: ", item_rating)
    return np.float16(item_rating)


def item_item_collaborative_filtering_pyspark(k, ratings, testDF):
    pivoted = ratings.groupBy("movieId").pivot('userId').sum('rating').na.fill(0)
    pivoted_T = ratings.groupBy("userId").pivot('movieId').sum('rating').na.fill(0)

    pivoted_dropped = pivoted_T.drop(pivoted_T.userId)
    columns = pivoted_dropped.columns
    mat = RowMatrix(pivoted_dropped.rdd.map(tuple))
    simsPerfect = mat.columnSimilarities()
    similarity_df = simsPerfect.entries.toDF()
    similarity_df = similarity_df.toDF(*['movie1', 'movie2', 'sim'])

    test_data = testDF.toPandas()
    item_item_collaborative_labels = [get_ratings_pyspark(
        x,
        pivoted,
        similarity_df,
        columns,
        k) for x in test_data[:].iterrows()]

    test_data['prediction_item_item_cf'] = item_item_collaborative_labels
    return test_data


def hybrid_calculation_function(rating_item_item, rating_als):
    if math.isnan(rating_als):
        return rating_item_item
    return (rating_item_item * 0.6) + (rating_als * 0.4)


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
    rmse = rmse_evaluator.evaluate(test_prediction)
    print("ALS RMSE: ", rmse)
    print("ALS RMSE: ", rmse, file=outFile)
    mae = mae_evaluator.evaluate(test_prediction)
    print("ALS MAE: ", mae)
    print("ALS MAE: ", mae, file=outFile)
    mse = mse_evaluator.evaluate(test_prediction)
    print("ALS MSE: ", mse)
    print("ALS MSE: ", mse, file=outFile)

    if cf:
        print("Running item-item collaborative filtering ...")
        time_start_cf = time.time()
        prediction_item_item = item_item_collaborative_filtering_pyspark(k, ratings, testDF)
        prediction_item_item_df = spark.createDataFrame(prediction_item_item)
        print("Running evaluations for item-item collaborative filtering ...")
        rmse = rmse_evaluator.evaluate(prediction_item_item_df)
        print("Item-Item CF RMSE: ", rmse)
        print("Item-Item CF RMSE: ", rmse, file=outFile)
        mae = mae_evaluator.evaluate(prediction_item_item_df)
        print("Item-Item CF MAE: ", mae)
        print("Item-Item CF MAE: ", mae, file=outFile)
        mse = mse_evaluator.evaluate(prediction_item_item_df)
        print("Item-Item CF MSE: ", mse)
        print("Item-Item CF MSE: ", mse, file=outFile)

        time_end = time.time()
        print("took {} minutes for item-item collaborative filtering.".format((time_end - time_start_cf) / 60))
        print("took {} minutes for item-item collaborative filtering.".format((time_end - time_start_cf) / 60),
              file=outFile)

        print("Combining ALS with item-item collaborative filtering ...")
        test_prediction_with_na = test_prediction_with_na.withColumnRenamed("prediction", "prediction_als")
        prediction_item_item_df = prediction_item_item_df.withColumnRenamed("prediction", "prediction_item_item_cf")

        prediction_total = prediction_item_item_df.join(test_prediction_with_na, ['userId', 'movieId', 'rating'])

        udf_hybrid_calc_function = F.udf(hybrid_calculation_function, DoubleType())
        prediction_total = prediction_total.withColumn("prediction",
                                                       udf_hybrid_calc_function("prediction_item_item_cf",
                                                                                "prediction_als"))

        print("Running evaluations for hybrid method...")
        rmse = rmse_evaluator.evaluate(prediction_total)
        print("Hybrid RMSE: ", rmse)
        print("Hybrid RMSE: ", rmse, file=outFile)
        mae = mae_evaluator.evaluate(prediction_total)
        print("Hybrid MAE: ", mae)
        print("Hybrid MAE: ", mae, file=outFile)
        mse = mse_evaluator.evaluate(prediction_total)
        print("Hybrid MSE: ", mse)
        print("Hybrid MSE: ", mse, file=outFile)

        if data_size >= 1000000:
            file_name = '../out/predictions_total-' + time_stamp + "-" + str(data_size) + '.csv'
            prediction_total.write.option("header", "true").csv(file_name)

        print("Recommending movies for 15 selected users: ")
        print("Recommending movies for 15 selected users: ", file=out_file)
        movies = spark.read.option("header", "true").csv('../data/ml-20m/movies.csv')
        unique_users = prediction_total.select('userId').distinct().limit(15).rdd.map(lambda r: r[0]).collect()
        for user in unique_users[:]:
            predictions_this_user = prediction_total.filter(prediction_total.userId == user)
            predictions_this_user = predictions_this_user.sort(prediction_total.prediction.desc())
            top_ten_recommendations = predictions_this_user.limit(10)
            top_ten_recommendations = top_ten_recommendations.select('movieId').rdd.map(lambda r: r[0]).collect()
            movies_filtered = movies.filter(movies.movieId.isin(top_ten_recommendations))
            print("Movie Recommendation for user {} :".format(user))
            movies_filtered.show()
            print("Movie Recommendation for user {} :".format(user), file=out_file)
            print(movies_filtered.toPandas().set_index("movieId"), file=out_file)

    print("Total time to run Script: {} minutes".format((time.time() - time_start) / 60))
    print("Total time to run Script: {} minutes".format((time.time() - time_start) / 60), file=outFile)
    spark.stop()


if __name__ == "__main__":
    # todo parameters
    data_size = 1000000  # 1000000 # 10000000  # total dataset is 20000263
    # for als if not cross validation
    rank = 1
    # cross validation
    crossValidation = True
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
