import random
import pandas as pd
from ID3 import *
import numpy as np
from math import sqrt


def calculate_centroid(example_set):
    centroid_list = []

    for i in range(len(example_set[0])):
        if i == 0:
            continue
        sum = 0
        for line in example_set:
            #print(line[i])
            sum += line[i]
        #print('---------------------')
        example_set_len = len(example_set)
        average = sum / example_set_len

        centroid_list.append(average)
    return centroid_list


def KNN_get_centroids_and_trees(N_param, example_set, feature_set):  # Learn N trees and return N centroids

    table_len = len(example_set)

    tree_and_centroid_list = []

    for i in range(N_param):
        example_set_tmp = example_set
        example_set_tmp = example_set_tmp.tolist()
        partial_example_set = []
        p = 0.3 + (random.random()*0.4)  # random number in the interval [0.3,0,7]
        partial_example_set_len = int(table_len * p)
        #print('partial_example_set_len: ', partial_example_set_len)

        for j in range(partial_example_set_len):
            choice = random.choice(example_set_tmp)
            partial_example_set.append(choice)
            example_set_tmp.remove(choice)

        #print('partial example_set: ', partial_example_set)
        centroid_list = calculate_centroid(partial_example_set)
        #print(centroid_list)

        tree = ID3(partial_example_set, feature_set, True, 0)

        new_tuple = (centroid_list, tree)
        tree_and_centroid_list.append(new_tuple)

    return tree_and_centroid_list


def first_element(element):
    return element[0]


def KNN_decision_tree(N_param, K_param):  # run_system() no inputs, the output is accuracy, make sure train.csv and test.csv in the project dir

    # todo Phase 1 : Create N trees and centroids in list: [(centroid,tree),...,(centroid,tree)]
    # read csv files
    file = pd.read_csv('train.csv')
    file2 = pd.read_csv('test.csv')

    # create DataFrames with pandas
    data_frame_train = pd.DataFrame(file)
    data_frame_test = pd.DataFrame(file2)

    data_frame_test_len = len(data_frame_test)
    feature_set = [element for element in data_frame_train]
    example_set = data_frame_train.to_numpy()

    centroid_and_tree_list = KNN_get_centroids_and_trees(N_param, example_set, feature_set)

    # todo Phase 2 : run test
    correct_answer = 0
    wrong_answers = 0
    test_set = data_frame_test.to_numpy()

    for i in range(data_frame_test_len):

        sample_vector = test_set[i][1:]           # create vector of all the features values

        # here we check distance between the single example centroid and the Trees centroids
        distance_and_tree_list = []
        for element in centroid_and_tree_list:        # creates list with Euclidean distances
            distance = calculate_distance(element[0], sample_vector)
            new_tuple = (distance, element[1])
            distance_and_tree_list.append(new_tuple)

        distance_and_tree_list.sort(key=first_element)         # sort by distance

        # here we send the single example to function that return the result of the committee
        single_example = data_frame_test.iloc[i:i + 1]
        total_result = check_single_example_on_K_trees(single_example, distance_and_tree_list, K_param)  # check the single example on K trees

        # here we check if the the Answer for our single example test is correct?
        real_diagnosis_bool = True if test_set[i][0] == 'M' else False
        if total_result == real_diagnosis_bool:
            correct_answer += 1
        else:
            wrong_answers += 1

    # here we check accuracy of our algorithm
    accuracy = correct_answer / data_frame_test_len
    print(accuracy)
    return accuracy


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
    total_result = True if M_counter >= B_counter else False
    return total_result


def calculate_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        print("not good")
        return 0
    sum = 0
    for i in range(len(vector1)):
        sum += (vector1[i]-vector2[i])*(vector1[i]-vector2[i])
    return sqrt(sum)








#KNN_decision_tree(1, 0)

# read csv files

file = pd.read_csv('train.csv')
file2 = pd.read_csv('test.csv')

# create DataFrames with pandas
data_frame = pd.DataFrame(file)
data_frame_test = pd.DataFrame(file2)

#feature_set = [element for element in data_frame]
#print(feature_set.index("perimeter_mean"))
se = data_frame.to_numpy()
for line in se:
    print(line)
    break
for line in se:
    line = line[1:]
    print(line)
    break

# data_frame = data_frame[data_frame["radius_mean"] > 25]
# se = data_frame.to_numpy()
# print(se)






