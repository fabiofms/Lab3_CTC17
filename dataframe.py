import math
import pandas as pd


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

    movies_df = pd.read_csv('movies_database.csv')
    ratings_df = pd.read_csv('ratings.dat', sep='::', engine='python', header=None,
                             names=['user_id', 'movie_id', 'classification', 'timestamp'])
    users_df = pd.read_csv('users.dat', sep='::', engine='python', header=None,
                           names=['user_id', 'gender', 'age', 'occupation', 'zip_code'])

    movies_df = movies_df[
        ['ID', 'Name', 'action', 'Gender', 'adventure', 'animation', 'children', 'comedy', 'crime', 'documentary', 'drama',
         'fantasy', 'fi', 'film', 'horror', 'musical', 'mystery', 'noir', 'romance', 'sci', 'thriller', 'war',
         'western']]
    movies_df.rename(columns={'ID': 'movie_id'}, inplace=True)
    # add year to movies
    movies_df['year'] = movies_df['Name'].apply(lambda x: (x.split(' ('))[-1])
    movies_df['year'] = movies_df['year'].apply(lambda x: (x.split(')'))[0])
    movies_df['year'] = movies_df['year'].apply(change_year)

    # use length of names
    movies_df['name_len'] = movies_df['Name'].apply(change_name)

    df = pd.merge(ratings_df, movies_df, on='movie_id', how='inner')
    df = pd.merge(df, users_df, on='user_id', how='inner')
    attributes_array = ['gender', 'age', 'occupation', 'action', 'adventure', 'animation', 'children',
                  'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'fi', 'film',
                  'horror', 'musical', 'mystery', 'noir', 'romance', 'sci', 'thriller',
                  'war', 'western', 'year', 'name_len']
    attributes = {}
    for a in attributes_array:
        attributes[a] = df[a].unique()

    # attributes = ['gender', 'age', 'occupation', 'Gender', 'year', 'name_len']
    pattern = df['classification'].value_counts().index[0]
    return df, attributes, pattern
