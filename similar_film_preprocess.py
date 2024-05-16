import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
        " ".join(x["keywords"])
        + " "
        + " ".join(x["cast"])
        + " "
        + x["director"]
        + " ".join(x["genres"])
    )
def preprocess(path):
    movies=pd.read_csv(path)
    features = ["actors", "tags","director", "genres"]
    for feature in features:
        movies[feature] = movies[feature].apply(literal_eval)
    for feature in features:
        movies[feature] = movies[feature].apply(clean_data)
    movies["all"] = movies.apply(create_all, axis=1)
    count = CountVectorizer()
    count_matrix = count.fit_transform(movies["all"])
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
    np.save('matrix',cosine_sim2)
    return

