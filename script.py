import Classifier
import dataframe

examples, attributes, pattern = dataframe.prepare()
tree = Classifier.decision_tree_learning(examples, attributes, pattern)

acertos = Classifier.predictor(tree, examples)
print('acertos: ', acertos)