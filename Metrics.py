import numpy as np
import pandas as pd
import math
from sklearn.feature_extraction.text import CountVectorizer


def get_confusion_matrix(prediction, y):
    prediction = np.array(prediction)
    y = np.array(y)
    m = [[0 for i in range(5)] for j in range(5)]
    m[0][0] = (np.where(np.logical_and(prediction == 1, y == 1), 1, 0)).sum()
    m[0][1] = (np.where(np.logical_and(prediction == 2, y == 1), 1, 0)).sum()
    m[0][2] = (np.where(np.logical_and(prediction == 3, y == 1), 1, 0)).sum()
    m[0][3] = (np.where(np.logical_and(prediction == 4, y == 1), 1, 0)).sum()
    m[0][4] = (np.where(np.logical_and(prediction == 5, y == 1), 1, 0)).sum()
    m[1][0] = (np.where(np.logical_and(prediction == 1, y == 2), 1, 0)).sum()
    m[1][1] = (np.where(np.logical_and(prediction == 2, y == 2), 1, 0)).sum()
    m[1][2] = (np.where(np.logical_and(prediction == 3, y == 2), 1, 0)).sum()
    m[1][3] = (np.where(np.logical_and(prediction == 4, y == 2), 1, 0)).sum()
    m[1][4] = (np.where(np.logical_and(prediction == 5, y == 2), 1, 0)).sum()
    m[2][0] = (np.where(np.logical_and(prediction == 1, y == 3), 1, 0)).sum()
    m[2][1] = (np.where(np.logical_and(prediction == 2, y == 3), 1, 0)).sum()
    m[2][2] = (np.where(np.logical_and(prediction == 3, y == 3), 1, 0)).sum()
    m[2][3] = (np.where(np.logical_and(prediction == 4, y == 3), 1, 0)).sum()
    m[2][4] = (np.where(np.logical_and(prediction == 5, y == 3), 1, 0)).sum()
    m[3][0] = (np.where(np.logical_and(prediction == 1, y == 4), 1, 0)).sum()
    m[3][1] = (np.where(np.logical_and(prediction == 2, y == 4), 1, 0)).sum()
    m[3][2] = (np.where(np.logical_and(prediction == 3, y == 4), 1, 0)).sum()
    m[3][3] = (np.where(np.logical_and(prediction == 4, y == 4), 1, 0)).sum()
    m[3][4] = (np.where(np.logical_and(prediction == 5, y == 4), 1, 0)).sum()
    m[4][0] = (np.where(np.logical_and(prediction == 1, y == 5), 1, 0)).sum()
    m[4][1] = (np.where(np.logical_and(prediction == 2, y == 5), 1, 0)).sum()
    m[4][2] = (np.where(np.logical_and(prediction == 3, y == 5), 1, 0)).sum()
    m[4][3] = (np.where(np.logical_and(prediction == 4, y == 5), 1, 0)).sum()
    m[4][4] = (np.where(np.logical_and(prediction == 5, y == 5), 1, 0)).sum()
    return m


def get_accuracy(prediction, y):
    df_aux = np.where(prediction == y, 1, 0)
    return df_aux.sum()/len(y)


def get_quadratic_error(prediction, y):
    return (sum((abs(prediction-y))**2))/len(y)


def get_kappa(acc, matrix):
    p0 = acc
    soma = [sum(matrix[i]) for i in range(len(matrix))]
    razoes = []
    cont = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            cont += matrix[j][i]
        razoes.append(cont)
    pe = 0
    for i in range(len(soma)):
        pe += soma[i]*(razoes[i]/sum(razoes))
    pe = pe/sum(soma)
    return (p0-pe)/(1-pe)


def get_specific_rows():
    def change_year(year_str):
        try:
            year = int(year_str)
            return 10 * math.trunc((year - 1900) / 10)
        except ValueError:
            return 0

    def change_name(name):
        if len(name.split(' ')) < 5:
            return 'short'
        else:
            return 'long'

    movies_df = pd.read_csv('./data/movies.dat', sep='::', engine='python', header=None,
                            names=['movie_id', 'title', 'genres'])

    # pre-processing movies in order to adapt gender information
    count = CountVectorizer()
    count_matrix = count.fit_transform(movies_df['genres'])
    count_vect_df = pd.DataFrame(count_matrix.todense(), columns=count.get_feature_names())
    movies_df = pd.concat([movies_df, count_vect_df], axis=1)

    # adding movie's year to attributes
    movies_df['year'] = movies_df['title'].apply(lambda x: (x.split(' ('))[-1])
    movies_df['year'] = movies_df['year'].apply(lambda x: (x.split(')'))[0])
    movies_df['year'] = movies_df['year'].apply(change_year)

    # using length of movie's names to attributes
    movies_df['name_len'] = movies_df['title'].apply(change_name)

    # filtering movies database
    movies = [1, 2, 356, 364, 480, 593, 837, 1644, 2167, 2364]
    db = movies_df.loc[movies_df['movie_id'].loc[movies], :]
    db['key'] = range(10)

    # merging to users database
    users_df = pd.DataFrame(data={'key': range(10), 'gender': ['M']*10, 'age': [18]*10, 'occupation': [4]*10})
    new_db = pd.merge(db, users_df, on = 'key')

    return new_db
