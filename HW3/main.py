# import packages

import pandas as pd
import copy
import random
from copy import deepcopy
from ID3 import ID3
from ID3 import Classifier





#feature_set = {element for element in df}
#print(df["radius_mean"])


#print(len(feature_set))

# # create dictionary from the big string
# dict1 = {}
# example_list = []
# feature_list = []
# lst = file.to_string().split(" ")
# lst = [x for x in lst if x != '']
# for e in lst:
#     if e == 'B' or e == 'M':
#         break
#     if e == 'concave':
#         continue
#
#     feature_list.append(e.replace("\n0", ''))
# counter = 0
# for i in range(len(lst)):
#     if lst[i] == 'M' or lst[i] == 'B':
#         j = i + 1
#         dict1[feature_list[0]] = lst[i]
#         while j < len(lst) and lst[j] != 'B' and lst[j] != 'M':
#             to_remove = '\n' + str(counter + 1)
#             dict1[feature_list[j - i]] = lst[j].replace(to_remove, '')
#             j += 1
#             # if lst[j] == 'M' or lst[j] == 'B':
#             #     object_list = lst[i:j-1]
#             #     i = j-1
#             #     break
#         example_list.append(deepcopy(dict1))
#         i = j - 1
#         counter += 1
#
# # print(object_list[0])
# # print(object_list[1])
# # print(object_list[2])
# feature_list.remove("diagnosis")
# return example_list, feature_list
#return 0,0


if __name__ == "__main__":

    # read csv file
    file = pd.read_csv('train.csv')
    file2 = pd.read_csv('test.csv')
    # create DataFrame with pandas
    data_frame = pd.DataFrame(file)
    data_frame_test = pd.DataFrame(file2)




    # data_frame = data_frame[data_frame["radius_mean"] > 25]
    # num = data_frame["radius_mean"].iloc[1]
    # print("there:",num)
    # for line in data_frame["radius_mean"]:
    #     if(line != num):
    #         print(line)
    # print(data_frame["radius_mean"])
    #newframe = data_frame.copy()
    #print(len(data_frame))

    # for i in range(len(data_frame)-1):
    #     first = data_frame.iloc[i:i + 2]["radius_mean"].iloc[0]
    #     second = data_frame.iloc[i:i + 2]["radius_mean"].iloc[1]
    #     threshold = (first + second) / 2
    #     print(first, second, threshold)

    #print(se, se.iloc[0],se.iloc[1])
    #print(data_frame)

    #lst = [i for i in range(len(data_frame))]
    #print(lst)
    #print(data_frame.reindex(lst))
    #print(data_frame)
    #print(data_frame.index)
    # print(data_frame["radius_mean"][14])
    #
    # print(data_frame.iloc[0:2]['radius_mean'].to_string()


    # for i in range(len(data_frame)-1):
    #     print("after: ", data_frame.iloc[i:i+2]["radius_mean"])








    # step 1 : training
    tree = ID3(data_frame, True)                   # main function

    # step 2 : testing
    test_real_diagnosis = [diagnosis for diagnosis in data_frame_test["diagnosis"]]
    correct_answer = 0
    wrong_answers = 0
    accuracy = 0
    for i in range(len(test_real_diagnosis)):
        result = Classifier(data_frame_test.loc[i], tree)
        #print("result", result)

        real_diagnosis_bool = True if test_real_diagnosis[i] == 'M' else False
        #print("real_diagnosis_bool", real_diagnosis_bool)
        #print("len(test_real_diagnosis)", len(test_real_diagnosis))
        if result == real_diagnosis_bool:
            correct_answer += 1
        else:
            wrong_answers += 1

    accuracy = correct_answer / len(test_real_diagnosis)
    print("accuracy: ", accuracy)










