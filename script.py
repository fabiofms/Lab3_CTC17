import Classifier
import dataframe

examples, attributes, pattern = dataframe.prepare()
# Classifier.get_predictor(train_set, attributes, pattern, min_gain, max_tree_depth)
predictor = Classifier.get_predictor(examples.iloc[1:1000], attributes, pattern, 0.005, 10)

acertos = predictor(examples.iloc[0:1000])
print('acertos: ', acertos)