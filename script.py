import DecisionTree
import ClassifierAPriori
import Dataframe
import Metrics
from sklearn.model_selection import train_test_split

# Preparing database and splitting the dataset in training and test sets
examples, attributes, pattern = Dataframe.prepare()
X = examples
y = examples['classification']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using decision tree algorithm to get prediction for movies
predictor = DecisionTree.get_predictor(X_train, attributes, pattern, 0.001, 6)
prediction_one = predictor(X_test)

# Using classifier a priori to get prediction for movies
predictor_apriori = ClassifierAPriori.ClassifierAPriori(X_train)
prediction_two = predictor_apriori.predict(X_test)

# Acurácia (taxa de acertos)
acc_one = Metrics.get_accuracy(prediction_one, y_test)
print('acuracia árvore de decisão: ', acc_one)
acc_two = Metrics.get_accuracy(prediction_two, y_test)
print('acuracia classificador a priori: ', acc_two)
print('')

# Matriz de confusão
print('matriz de confusão árvore de decisão: ')
m = Metrics.get_confusion_matrix(prediction_one, y_test)
for line in m:
    print(line)
print('matriz de confusão classificador a priori: ')
n = Metrics.get_confusion_matrix(prediction_two, y_test)
for line in n:
    print(line)
print('')

# Erro quadrático
print('erro quadrático médio árvore de decisão: ', Metrics.get_quadratic_error(prediction_one, y_test))
print('erro quadrático médio classificador a priori: ', Metrics.get_quadratic_error(prediction_two, y_test))
print('')

# Kappa
print('kappa árvore de decisão: ', Metrics.get_kappa(acc_one, m))
print('kappa classificador a priori: ', Metrics.get_kappa(acc_two, n))
print('')

# Specific ratings
print('ratings para árvore de decisão: ', print(predictor(Metrics.get_specific_rows())))
print('ratings para classificador a priori: ', print(predictor_apriori.predict([1, 2, 356, 364, 480, 593, 837, 1644,
                                                                                2167, 2364])))
print('')

input()
