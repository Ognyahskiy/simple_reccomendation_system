import pandas as pd
import sys

from surprise import dump

path,userId,algo=sys.argv[1],int(sys.argv[2]),sys.argv[3]
ratings = pd.read_csv(path)
pred,algo=dump.load(algo)
all_movies = set(ratings['movieId'].unique())
rated_movies = set(ratings[ratings['userId'] == userId]['movieId'].unique())
unrated_movies = list(all_movies - rated_movies)
predictions = [algo.predict(userId, movieId) for movieId in unrated_movies]
top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:10]
prediction=[]
for pred in top_predictions:
    prediction.append(pred.iid)
print(prediction)
sys.stdout.flush()