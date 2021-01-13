import random
import pandas as pd
from ID3 import *
import numpy as np
from math import sqrt

def KNN_decision_tree(N_param, K_parameter, data_frame):  # 102 to 240

    table_len = len(data_frame["diagnosis"])

    tree_and_centroid_list = []

    for i in range(N_param):
        number_list = [i for i in range(table_len)]
        index_list = []
        p = 0.3 + (random.random()*0.4)  # random number in the interval [0.3,0,7]
        partial_example_set_len = int(table_len * p)
        #print(partial_example_set_len)
        for j in range(partial_example_set_len):
            index = random.choice(number_list)
            index_list.append(index)
            number_list.remove(index)
        partial_example_df = data_frame.reindex(index_list)     # partial example set
        print(partial_example_df)

        centroid_list = calculate_centriod(partial_example_df)
        tree = ID3(partial_example_df, True, 0)
        new_tuple = (centroid_list, tree)
        tree_and_centroid_list.append(new_tuple)
    return tree_and_centroid_list


def calculate_centriod(data_frame):
    centroid_list = []
    for feature in data_frame:
        if feature == "diagnosis":
            continue
        arr = data_frame[feature].to_numpy()
        sum = np.sum(arr)
        data_frame_len = len(data_frame)
        average = sum / data_frame_len

        centroid_list.append(average)
    return centroid_list


def second_element(element):
    return element[1]


def run_KNN_system(N_param, K_param):  # run_system() no inputs, the output is accuracy, make sure train.csv and test.csv in the project dir


    # read csv files
    file = pd.read_csv('train.csv')
    file2 = pd.read_csv('test.csv')

    # create DataFrames with pandas
    data_frame = pd.DataFrame(file)
    data_frame_test = pd.DataFrame(file2)



    tree_and_centroid_list = KNN_decision_tree(N_param, K_param, data_frame)


    # todo run test
    test_real_diagnosis = [diagnosis for diagnosis in data_frame_test["diagnosis"]]
    correct_answer = 0
    wrong_answers = 0
    for i in range(len(test_real_diagnosis)):

        sample_vector = []
        to_test = data_frame_test.iloc[i:i + 1]   # row in the test table that we need to test

        for feature in data_frame_test:        # create vector of all the features values
            if feature == "diagnosis":
                continue
            sample_vector.append(to_test[feature].iloc[0])

        distance_list = []
        for element in tree_and_centroid_list:        # creates list with Euclidean distances
            distance = calculate_distance(element[0], sample_vector)
            new_tuple = (distance, element[1])
            distance_list.append(new_tuple)

        distance_list.sort(key=second_element)

        M_counter = 0
        B_counter = 0
        for s in range(K_param):                     # check the sample in K trees
            tree = distance_list[s][1]
            result = Classifier(to_test, tree)
            if result is True:
                M_counter += 1
            elif result is False:
                B_counter += 1
        total_result = True if M_counter >= B_counter else False

        real_diagnosis_bool = True if test_real_diagnosis[i] == 'M' else False
        if total_result == real_diagnosis_bool:
            correct_answer += 1
        else:
            wrong_answers += 1

    accuracy = correct_answer / len(test_real_diagnosis)
    return accuracy


def calculate_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        print("not good")
        return 0
    sum = 0
    for i in range(len(vector1)):
        sum += (vector1[i]-vector2[i])*(vector1[i]-vector2[i])
    return sqrt(sum)








#run_KNN_system(1,0)

# read csv files
file = pd.read_csv('train.csv')
file2 = pd.read_csv('test.csv')

# create DataFrames with pandas
data_frame = pd.DataFrame(file)
data_frame_test = pd.DataFrame(file2)


#se = data_frame["radius_mean"]
#arr = se.to_numpy()
#print(np.sum(arr))

arr = np.array([], dtype='int16')
arr = np.append(arr, 21)
arr = np.append(arr, 25)
print(arr)
print(data_frame_test["radius_mean"].iloc[0:1].iloc[0])

