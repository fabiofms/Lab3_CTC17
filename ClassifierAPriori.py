
class ClassifierAPriori:
    def __init__(self, db):
        self.classification, self.movies = self.get_recommendation(db)

    def get_recommendation(self, db):
        movies = db['movie_id'].unique()
        mean = []
        movie = []
        for mov in movies:
            ratings = db.loc[db.index[db['movie_id'] == mov], 'classification']
            movie.append(mov)
            mean.append(int(sum(ratings) / len(ratings)))
        return mean, movie

    def predict(self, x):
        ans = []
        x = x['movie_id']
        for elem in x:
            if elem in self.movies:
                ans.append(self.classification[self.movies.index(elem)])
            else:
                ans.append(5)
        return ans
