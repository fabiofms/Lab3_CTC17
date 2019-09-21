import math
import pandas as pd
import numpy as np


def get_predictor(examples_complete, attributes_complete, pattern_complete, min_gain=0, max_depth=1000000):

    def decision_tree_learning(examples, attributes, pattern):
        # examples argument is a pandas data frame
        # attributes is a list of columns labels we're able to use
        # pattern is a classification to use in certain situations

        # nested classes
        class Node:
            def __init__(self, type_argument, attribute):
                if type_argument == 'leaf':
                    self.classification = attribute
                    self.leaf = True
                elif type_argument == 'internal':
                    self.attribute = attribute
                    self.children = {}
                    self.leaf = False

            def add(self, new_value, new_node):
                self.children[new_value] = new_node

            def set_leaf(self):
                if self.leaf:
                    self.leaf = False
                else:
                    self.leaf = True

            def is_leaf(self):
                return self.leaf

            def set_classification(self, classification):
                self.classification = classification

            def get_classification(self):
                return self.classification

            def get_attribute(self):
                return self.attribute

            def get_node(self, value):
                return self.children[value]

        # nested functions
        def same_classification(df):
            return len(df['classification'].unique()) == 1

        def pop_classification(df):
            return df['classification'].iloc[0]

        def majority_value(df):
            return df['classification'].value_counts().index[0]

        def get_values(attribute, df):
            return df[attribute].unique()

        def subset(subset_df, subset_value, subset_best):
            return subset_df[subset_df[subset_best] == subset_value]

        def choose_attribute(attributes2choose, examples2use):
            # nested functions
            def entropy(examples_subset):
                base = len(examples['classification'].unique())  # number of classifications available
                total_value = len(examples_subset)
                values_frequency = examples_subset['classification'].value_counts().apply(
                    lambda x: (-x / total_value) * math.log(x / total_value, base))
                return values_frequency.sum()

            def gain(attribute):
                result = 0
                total_len = len(examples2use)
                for v in get_values(attribute, examples2use):
                    subset = examples2use[examples2use[attribute] == v]
                    subset_len = len(subset)
                    result += entropy(subset) * subset_len / total_len
                return entropy(examples2use) - result

            attribute_result = attributes2choose[0]
            attribute_result_gain = gain(attribute_result)
            for attribute in attributes2choose:
                if gain(attribute) > attribute_result_gain:
                    attribute_result = attribute
                    attribute_result_gain = gain(attribute)

            return attribute_result, attribute_result_gain

        # Algorithm
        # print('Rodou algoritmo')
        # Stop conditions of recursion
        if examples.empty:
            return Node('leaf', pattern)
        elif same_classification(examples):
            # print('len of examples: ', len(examples))
            return Node('leaf', pop_classification(examples))
        elif len(attributes) <= max([len(attributes_complete) - max_depth, 0]):  # threshold level for depth
            return Node('leaf', majority_value(examples))
        else:
            # best is a label
            best, gain = choose_attribute(attributes, examples)
            # threshold level for gain
            if gain < min_gain:
                # print('low gain: ', gain)
                return Node('leaf', majority_value(examples))
            # print('best is: ' + best)
            # tree is the data structure of the decision tree
            tree = Node('internal', best)
            m = majority_value(examples)
            # print('values: ', get_values(best, examples))
            for value in get_values(best, examples):
                examples_temp = subset(examples, value, best)  # add best argument in order to keep function pure
                attributes_temp = attributes.copy()
                attributes_temp.remove(best)
                # print('attributes_temp: ', attributes_temp)
                # print('m: ', m)
                # print('value: ', value)
                new_tree = decision_tree_learning(examples_temp, attributes_temp, m)
                tree.add(value, new_tree)
        return tree

    def predictor(tree):
        def predict_row(row):
            node = tree
            while not node.is_leaf():
                attribute = node.get_attribute()
                value = row[attribute]
                node = node.get_node(value)
            return node.get_classification()

        def predict_df(df):
            df_aux = df.apply(predict_row, axis=1)
            df_aux = np.where(df_aux == df['classification'], 1, 0)
            return df_aux.sum()

        return predict_df

    tree = decision_tree_learning(examples_complete, attributes_complete, pattern_complete)
    return predictor(tree)
