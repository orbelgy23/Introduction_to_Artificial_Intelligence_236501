# import packages

import pandas as pd
import copy
import random
from copy import deepcopy
from ID3 import ID3
from ID3 import Classifier

if __name__ == "__main__":

    # read csv file
    file = pd.read_csv('train.csv')
    file2 = pd.read_csv('test.csv')
    # create DataFrame with pandas
    data_frame = pd.DataFrame(file)
    data_frame_test = pd.DataFrame(file2)

    # todo step 1 : training
    tree = ID3(data_frame, True)                   # main function

    # todo step 2 : testing
    test_real_diagnosis = [diagnosis for diagnosis in data_frame_test["diagnosis"]]
    correct_answer = 0
    wrong_answers = 0
    accuracy = 0
    for i in range(len(test_real_diagnosis)):
        result = Classifier(data_frame_test.loc[i], tree)

        real_diagnosis_bool = True if test_real_diagnosis[i] == 'M' else False
        #print("real_diagnosis_bool", real_diagnosis_bool)
        #print("len(test_real_diagnosis)", len(test_real_diagnosis))
        if result == real_diagnosis_bool:
            correct_answer += 1
        else:
            wrong_answers += 1

    accuracy = correct_answer / len(test_real_diagnosis)
    print("accuracy: ", accuracy)










