import random
import pandas as pd
from ID3 import *
import numpy as np
from math import sqrt


def first_element(element):
    return element[0]


def KNN_decision_tree(N_param, K_param):

    # todo Phase 1 : Creates N tuples of trees and centroids in list: [(centroid,tree),...,(centroid,tree)]
    # read csv files
    file = pd.read_csv('train.csv')
    file2 = pd.read_csv('test.csv')

    # create DataFrames with pandas
    data_frame_train = pd.DataFrame(file)
    data_frame_test = pd.DataFrame(file2)

    data_frame_test_len = len(data_frame_test)
    feature_set = [element for element in data_frame_train]
    example_set = data_frame_train.to_numpy()

    centroid_and_tree_list = KNN_get_centroids_and_trees(N_param=N_param, example_set=example_set, feature_set=feature_set)

    # todo Phase 2 : run test
    correct_answer = 0
    wrong_answers = 0
    test_set = data_frame_test.to_numpy()

    for i in range(data_frame_test_len):

        sample_vector = test_set[i][1:]           # given new example -> create vector of all the features values

        # here we check distance between the single example centroid and the trees centroids
        distance_and_tree_list = []
        for element in centroid_and_tree_list:        # creates list with Euclidean distances
            distance = calculate_distance(element[0], sample_vector)
            new_tuple = (distance, element[1])
            distance_and_tree_list.append(new_tuple)

        distance_and_tree_list.sort(key=first_element)         # sort by distance

        # here we send the single example to function that returns the result of the committee
        single_example = data_frame_test.iloc[i:i + 1]
        total_result = check_single_example_on_K_trees(single_example, distance_and_tree_list, K_param)  # check the single example on K trees

        # here we check if the the answer for our single example is correct?
        real_diagnosis_bool = True if test_set[i][0] == 'M' else False
        if total_result == real_diagnosis_bool:
            correct_answer += 1
        else:
            wrong_answers += 1

    # here we check accuracy of our algorithm
    accuracy = correct_answer / data_frame_test_len
    print(accuracy)
    return accuracy


def KNN_get_centroids_and_trees(N_param, example_set, feature_set):  # Learn N trees and return N centroids

    table_len = len(example_set)

    tree_and_centroid_list = []

    p = 0.3 + (random.random() * 0.4)  # random number in the interval [0.3,0,7]
    partial_example_set_len = int(table_len * p)

    for i in range(N_param):
        example_set_tmp = example_set
        example_set_tmp = example_set_tmp.tolist()
        partial_example_set = []

        for j in range(partial_example_set_len):       # selects subset of the example set
            choice = random.choice(example_set_tmp)
            partial_example_set.append(choice)
            example_set_tmp.remove(choice)

        centroid_list = calculate_centroid(partial_example_set)   # calculate centroid

        tree = ID3(example_set=partial_example_set, feature_set=feature_set, classification=True, m_param=0)

        new_tuple = (centroid_list, tree)
        tree_and_centroid_list.append(new_tuple)

    return tree_and_centroid_list


def calculate_centroid(example_set):
    centroid_list = []

    for i in range(len(example_set[0])):
        if i == 0:
            continue

        sum_ = 0
        for line in example_set:
            sum_ += line[i]

        example_set_len = len(example_set)
        average = sum_ / example_set_len

        centroid_list.append(average)
    return centroid_list


def calculate_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        print("The dimensions of the vectors are not equal")
        return -1
    sum_ = 0
    for i in range(len(vector1)):
        sum_ += (vector1[i]-vector2[i])*(vector1[i]-vector2[i])
    return sqrt(sum_)


def check_single_example_on_K_trees(single_example, distances_and_trees, K_param):
    M_counter = 0
    B_counter = 0
    for s in range(K_param):  # check the sample in K trees
        tree = distances_and_trees[s][1]
        result = Classifier(single_example, tree)
        if result is True:
            M_counter += 1
        else:
            B_counter += 1
    total_result = True if M_counter > B_counter else False
    return total_result


def KNN_decision_tree_experiment(N_param):        # check for the best N and K parameters

    max_avg = 0
    max_N, max_K = 0, 0
    max_accuracy = 0
    for i in range(1, N_param + 1):
        for j in range(1, i + 1):
            accuracy_lst = []
            sum = 0
            print("calculate for N = " + str(i) + " K = " + str(j))
            for k in range(5):
                accuracy_lst.append(KNN_decision_tree(i, j))
            for e in accuracy_lst:
                sum += e
            avg = sum / 5
            if avg > max_avg:
                max_avg = avg
                max_N, max_K = i, j
                max_accuracy = max(accuracy_lst)
    print("best N = " + str(max_N) + " best K = " + str(max_K) + " best accuracy for this parameters: " + str(max_accuracy))
