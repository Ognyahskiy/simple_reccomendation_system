import os

import pandas as pd
import numpy as np
from surprise import Dataset, Reader,SVD,dump
from flask import Flask, jsonify,request, send_file
from dotenv import load_dotenv
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

app = Flask(__name__)

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ""
            
def create_all(x):
    return (
        " ".join(x["tags"])
        + " ".join(x["genre"])
    )


@app.route('/similar_preprocess', methods=['GET', 'POST'], endpoint='preprocess')  # метод авторизации пользователя
def preprocess():
    path=os.getenv('MOVIES')
    movies=pd.read_csv(path)
    features = ["tags","genre"]
    for feature in features:
        movies[feature] = movies[feature].apply(literal_eval)
    features = ["tags","genre"]
    for feature in features:
        movies[feature] = movies[feature].apply(clean_data)
    movies["all"] = movies.apply(create_all, axis=1)
    count = CountVectorizer()
    count_matrix = count.fit_transform(movies["all"])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    np.save('matrix',cosine_sim)
    return "Done preprocessing movies"

@app.route('/similar_ids', methods=['POST'], endpoint='get_recommendations')
def get_recommendations():
    data=request.json
    id=data.get('id')
    path=os.getenv('MOVIES')
    cosine_sim=np.load(os.getenv('SIMILAR_MATRIX'))
    movies = pd.read_csv(path)
    movies = movies.reset_index()
    indices = pd.Series(movies.index, index=movies["id"])
    idx = indices[id]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return jsonify(movies['id'].iloc[movie_indices].values.tolist())

@app.route('/recommend_preprocess', methods=['GET'], endpoint='recommend_prep')
def recomend_prep():
    path=os.getenv('RATINGS')
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
    return "Algo done"

@app.route('/recommend', methods=['POST'], endpoint='recommend')
def reccomend_id():
    path=os.getenv('RATINGS')
    algo=os.getenv('RECOMMEND_ALGO')
    data=request.json
    userId=data.get('id')
    ratings = pd.read_csv(path)
    pred,algo=dump.load(algo)
    all_movies = set(ratings['movieId'].unique())
    rated_movies = set(ratings[ratings['userId'] == userId]['movieId'].unique())
    unrated_movies = list(all_movies - rated_movies)
    predictions = [algo.predict(userId, movieId) for movieId in unrated_movies]
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:10]
    prediction=[]
    for pred in top_predictions:
        prediction.append(int(pred.iid))
    return jsonify(prediction)

app.run(host='0.0.0.0', port=3000)
