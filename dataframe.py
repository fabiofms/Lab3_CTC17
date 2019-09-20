import pandas as pd


def prepare():

    movies_df = pd.read_csv('movies.dat', sep='::', engine='python', header=None,
                            names=['movie_id', 'title', 'genres'])
    ratings_df = pd.read_csv('ratings.dat', sep='::', engine='python', header=None,
                             names=['user_id', 'movie_id', 'classification', 'timestamp'])
    users_df = pd.read_csv('users.dat', sep='::', engine='python', header=None,
                           names=['user_id', 'gender', 'age', 'occupation', 'zip_code'])


    df = pd.merge(ratings_df, movies_df, on='movie_id', how='inner')
    df = pd.merge(df, users_df, on='user_id', how='inner')
    attributes = ['gender', 'age', 'occupation']

    return df, attributes, 3

