import Classifier
import dataframe
import numpy as np
from sklearn.model_selection import train_test_split

examples, attributes, pattern = dataframe.prepare()
X = examples
y = examples['classification']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classifier.get_predictor(train_set, attributes, pattern, min_gain, max_tree_depth)
predictor = Classifier.get_predictor(X_train, attributes, pattern, 0.005, 10)

prediction = predictor(X_test)
df_aux = np.where(prediction == y_test, 1, 0)
acertos = df_aux.sum()
acuracia = acertos/len(y_test)
print('acuracia: ', acuracia)
