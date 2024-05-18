import pandas as pd
import sys

from surprise import Dataset, Reader,SVD,dump

path=sys.argv[1]
ratings = pd.read_csv(path)
user_mean_ratings = ratings.groupby('userId')['rating'].mean()
ratings = pd.merge(ratings, user_mean_ratings.rename('user_mean_rating'), left_on='userId', right_index=True)
ratings['rating'] = ratings['rating'] - ratings['user_mean_rating']
reader = Reader(rating_scale=(-5,5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
algo = SVD()
trainset = data.build_full_trainset()
algo.fit(trainset)
dump.dump('algo',algo=algo)
