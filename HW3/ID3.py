from math import log
import pandas as pd
import random
from copy import deepcopy


class Tree:

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.classification = None
        self.leftNode = None
        self.rightNode = None


def ID3(data_frame, classification):
    tree = Tree()
    if len(data_frame["diagnosis"]) <= 0:
        tree.classification = classification
        return tree

    classification = MajorityClass(data_frame)

    if isLeaf(data_frame) is True:
        tree.classification = classification
        return tree



    feature_set = [element for element in data_frame]
    #feature_set_tmp = {'perimeter_mean', 'perimeter_worst', 'texture_mean', 'concavity_mean', 'symmetry_mean', 'area_se', 'compactness_mean', 'area_worst', 'smoothness_worst', 'smoothness_mean', 'texture_se', 'concavity_worst', 'symmetry_worst', 'fractal_dimension_se', 'texture_worst', 'concave points_se', 'smoothness_se', 'concavity_se', 'fractal_dimension_worst', 'perimeter_se', 'radius_worst', 'concave points_worst', 'compactness_worst', 'diagnosis', 'compactness_se', 'fractal_dimension_mean', 'concave points_mean', 'symmetry_se', 'area_mean', 'radius_mean', 'radius_se'}
    #feature_set_tmp = {'perimeter_mean', 'perimeter_worst'}
    #feature_set_tmp = {'radius_mean', 'symmetry_worst', 'texture_mean', 'area_mean', 'perimeter_worst'}
    #print('before: \n', data_frame)
    best_feature, best_threshold = MaxIG(data_frame, feature_set)        # Select the best feature
    #print('after: \n', data_frame)
    #print(best_feature, best_threshold) # 0.9380530973451328

    if all_values_equal(data_frame, best_feature):
        tree.classification = classification
        return tree

    smaller_equal_df = data_frame.copy()
    bigger_df = data_frame.copy()

    #print(best_feature, best_threshold)

    smaller_equal_df = smaller_equal_df[smaller_equal_df[best_feature] < best_threshold]  # create data frame with people with feature <= threshold
    bigger_df = bigger_df[bigger_df[best_feature] >= best_threshold]  # create data frame with people with feature > threshold

    #print("smaller equal df: \n", smaller_equal_df)
    #print("bigger df: \n", bigger_df)

    tree.feature = best_feature
    tree.threshold = best_threshold

    tree.leftNode = ID3(smaller_equal_df, classification)

    tree.rightNode = ID3(bigger_df, classification)

    # print(smaller_equal_df)
    # print(bigger_df)
    # print('best_feature:', best_feature)

    return tree


def MaxIG(data_frame, feature_set):            # calculate the feature that has most IG
    max_value = -1
    max_value_feature = ""
    best_threshold = -1
    for feature in feature_set:
        if feature == "diagnosis":
            continue

        value, threshold = IG(data_frame, feature)
        #print('value: ', value, " | feature: ", feature, " | threshold: ", threshold)
        if value > max_value:
            max_value = value
            max_value_feature = feature
            best_threshold = threshold
    return max_value_feature, best_threshold


def IG(data_frame, feature):      # calculate the max threshold for certain feature
    #print(feature)
    ig_max_value = -1
    ig_max_threshold = -1
    current_ig = calculate_entropy(data_frame)
#    print('current_ig: ', current_ig)
    for i in range(len(data_frame)-1):
    #for i in range(2):
        #print(data_frame[feature][i], data_frame[feature][i+1])
        #print(len(data_frame) - 1)
        #threshold = 10
        first = data_frame.iloc[i:i+2][feature].iloc[0]
        second = data_frame.iloc[i:i+2][feature].iloc[1]
        #print(first, second)
        threshold = (first + second)/2
        #threshold = (data_frame[feature][i] + data_frame[feature][i+1])/2
        #print("threshold", threshold)
        ig_value = calculate_IG_for_threshold(data_frame, threshold, feature)
        ig_diff = current_ig - ig_value
        #print(ig_value)
        if(ig_diff > ig_max_value):
            ig_max_value = ig_diff
            ig_max_threshold = threshold
    return ig_max_value, ig_max_threshold


def calculate_IG_for_threshold(data_frame, threshold, feature):         # calculate entropy for certain threshold for certain feature
    #print(threshold)
    total_len = len(data_frame["diagnosis"])

    smaller_equal_df = data_frame.copy()
    bigger_df = data_frame.copy()

    smaller_equal_df = smaller_equal_df[smaller_equal_df[feature] < threshold]  # create data frame with people with feature <= threshold
    bigger_df = bigger_df[bigger_df[feature] >= threshold]  # create data frame with people with feature > threshold

    smaller_equal_df_len = len(smaller_equal_df)
    bigger_df_len = len(bigger_df)

    return (smaller_equal_df_len/total_len) * calculate_entropy(smaller_equal_df) + (bigger_df_len/total_len) * calculate_entropy(bigger_df)


def calculate_entropy(data_frame):
    total_len = len(data_frame["diagnosis"])

    if total_len == 0:
        return 0

    total_M = 0
    total_B = 0
    for line in data_frame["diagnosis"]:
        if line == 'M':
            total_M += 1
        elif line == 'B':
            total_B += 1
    B_percentage = total_B / total_len
    M_percentage = total_M / total_len
    if B_percentage == 0:
        return (-1) * M_percentage * log(M_percentage, 2)
    if M_percentage == 0:
        return (-1) * B_percentage * log(B_percentage, 2)
    num1 = B_percentage * log(B_percentage, 2)
    num2 = M_percentage * log(M_percentage, 2)
    return (-1)*(num1+num2)


def isLeaf(data_frame):
    M_counter = 0
    B_counter = 0
    for line in data_frame["diagnosis"]:
        if line == 'M':
            M_counter += 1
        elif line == 'B':
            B_counter += 1
    if M_counter == 0 or B_counter == 0:
        return True
    return False


def MajorityClass(data_frame):      # True if |M| >= |B| , False if |M| < |B|
    M_counter = 0
    B_counter = 0
    for line in data_frame["diagnosis"]:
        if line == 'M':
            M_counter += 1
        elif line == 'B':
            B_counter += 1
    if M_counter >= B_counter:
        return True
    else:
        return False


def Classifier(data_frame, tree):
    if tree.leftNode is None and tree.rightNode is None:
        return tree.classification

    current_feature = tree.feature
    current_threshold = tree.threshold
    if data_frame[current_feature] < current_threshold:
        return Classifier(data_frame, tree.leftNode)
    else:
        return Classifier(data_frame, tree.rightNode)

def all_values_equal(data_frame,feature):
    first = data_frame[feature].iloc[0]
    for line in data_frame[feature]:
        if(line != first):
            return False
    return True











