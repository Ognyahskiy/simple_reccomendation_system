import numpy as np
import pandas as pd
import sys

def get_recommendations(path,id, cosine_sim):
    movies = pd.read_csv(path)
    movies = movies.reset_index()
    indices = pd.Series(movies.index, index=movies["id"])
    idx = indices[id]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return movie_indices

path,id,cosine_sim=sys.argv[1],int(sys.argv[2]),np.load(sys.argv[3])
films=get_recommendations(path,id,cosine_sim)
print(films)
sys.stdout.flush()
