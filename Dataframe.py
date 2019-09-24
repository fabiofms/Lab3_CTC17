# importing libraries
import math
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# function designed to get the dataframe ready
def prepare():
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

    # importing databases
    movies_df = pd.read_csv('./data/movies.dat', sep='::', engine='python', header=None,
                            names=['movie_id', 'title', 'genres'])
    ratings_df = pd.read_csv('./data/ratings.dat', sep='::', engine='python', header=None,
                             names=['user_id', 'movie_id', 'classification', 'timestamp'])
    users_df = pd.read_csv('./data/users.dat', sep='::', engine='python', header=None,
                           names=['user_id', 'gender', 'age', 'occupation', 'zip_code'])

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

    # merging all three dataframes initially imported
    df = pd.merge(ratings_df, movies_df, on='movie_id', how='inner')
    df = pd.merge(df, users_df, on='user_id', how='inner')

    # getting unique values for each attribute
    attributes_array = ['gender', 'age', 'occupation', 'action', 'adventure', 'animation', 'children',
                  'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'fi', 'film',
                  'horror', 'musical', 'mystery', 'noir', 'romance', 'sci', 'thriller',
                  'war', 'western', 'year', 'name_len']
    attributes = {}
    for a in attributes_array:
        attributes[a] = df[a].unique()

    # getting class with most appearances
    pattern = df['classification'].value_counts().index[0]

    # returning data frame, unique values for attributes and for target
    return df, attributes, pattern
