from math import log
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from timeit import default_timer as timer


class Tree:

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.classification = None
        self.leftNode = None
        self.rightNode = None

        self.true_childs = None
        self.false_childs = None


def calculate_true_or_false_childs(tree):
    # if tree.rightNode is None or tree.leftNode is None:
    #     if tree.rightNode is None:
    #         return tree.leftNode.true_childs, tree.leftNode.false_childs
    #     elif tree.leftNode is None:
    #         return tree.rightNode.true_childs, tree.rightNode.false_childs

    # else:
    return tree.rightNode.true_childs + tree.leftNode.true_childs, tree.rightNode.false_childs + tree.leftNode.false_childs


def ID3(example_set, feature_set, classification, m_param):
    tree = Tree()
    # print(len(example_set))

    if len(example_set) <= 0:
        tree.classification = classification
        tree.true_childs = 1 if classification is True else 0
        tree.false_childs = 1 if classification is False else 0
        return tree



    if len(example_set) < m_param:
        tree.classification = classification
        tree.true_childs = 1 if classification is True else 0
        tree.false_childs = 1 if classification is False else 0
        return tree

    classification = MajorityClass(example_set)

    if isLeaf(example_set) is True:
        tree.classification = classification
        tree.true_childs = 1 if classification is True else 0
        tree.false_childs = 1 if classification is False else 0
        return tree

    best_feature_index, best_threshold, best_feature_name = MaxIG(example_set, feature_set)        # Select the best feature

    #print("best feature: ", best_feature_index, "best threshold: ", best_threshold)


    if all_values_equal(example_set, best_feature_index):
        tree.classification = classification
        tree.true_childs = 1 if classification is True else 0
        tree.false_childs = 1 if classification is False else 0
        return tree


    #print(best_feature, best_threshold)

    smaller = []
    bigger_equal = []
    for line in example_set:
        if line[best_feature_index] < best_threshold:
            smaller.append(line)
        else:
            bigger_equal.append(line)

    tree.feature = best_feature_name

    tree.threshold = best_threshold

    tree.leftNode = ID3(smaller, feature_set, classification, m_param)

    tree.rightNode = ID3(bigger_equal, feature_set, classification, m_param)

    tree.true_childs, tree.false_childs = calculate_true_or_false_childs(tree)

    return tree


def MaxIG(example_set, feature_set):            # calculate the feature that has most IG
    max_value = -1
    max_value_feature_index = -1
    max_value_feature_name = ""
    best_threshold = -1
    for feature in feature_set:
        if feature == "diagnosis":
            continue

        curr_feature_index = feature_set.index(feature)
        # recent changes:
        # new_df = data_frame[["diagnosis", feature]]
        # arr = new_df.to_numpy()
        value, threshold = IG(example_set, curr_feature_index)

        # value, threshold = IG(data_frame, feature)
        # print('value: ', value, " | feature: ", feature, " | threshold: ", threshold)
        if value >= max_value:
            max_value = value
            max_value_feature_index = curr_feature_index
            max_value_feature_name = feature
            best_threshold = threshold
    return max_value_feature_index, best_threshold, max_value_feature_name


def get_i_element(line, i):
    return line[i]


def IG(arr, feature_index):      # calculate the max threshold for certain feature

    arr_len = len(arr)

    ig_max_value = -1
    ig_max_threshold = -1
    current_ig = calculate_entropy(arr)

    new_lst = []

    for line in arr:
        new_lst.append(line[feature_index])

    new_lst.sort()

    for i in range(arr_len - 1):




        #arr.sort(key=get_i_element(arr[i]))

        sum = new_lst[i] + new_lst[i + 1]
        if new_lst[i] == new_lst[i + 1]:
            continue
        threshold = sum / 2
        ig_value = calculate_IG_for_threshold(arr, feature_index, threshold, arr_len)  # passing arr_len to save run time
        ig_diff = current_ig - ig_value
        if ig_diff > ig_max_value:
            ig_max_value = ig_diff
            ig_max_threshold = threshold
    return ig_max_value, ig_max_threshold




def calculate_IG_for_threshold(arr, feature_index, threshold, arr_len):         # calculate entropy for certain threshold for certain feature
    smaller = []
    bigger_equal = []


    for i in range(arr_len):
        if arr[i][feature_index] >= threshold:
            bigger_equal.append(arr[i])
        else:
            smaller.append(arr[i])

    return (len(smaller)/arr_len) * calculate_entropy(smaller) + (len(bigger_equal)/arr_len) * calculate_entropy(bigger_equal)


def calculate_entropy(arr):

    arr_len = len(arr)
    if arr_len == 0:
        return 0

    total_M = 0
    total_B = 0
    for line in arr:
        if line[0] == 'M':
            total_M += 1
        else:
            total_B += 1
    B_percentage = total_B / arr_len
    M_percentage = total_M / arr_len
    if B_percentage == 0:
        return (-1) * M_percentage * log(M_percentage, 2)
    if M_percentage == 0:
        return (-1) * B_percentage * log(B_percentage, 2)
    num1 = B_percentage * log(B_percentage, 2)
    num2 = M_percentage * log(M_percentage, 2)
    return (-1)*(num1+num2)


def isLeaf(example_set):
    M_counter = 0
    B_counter = 0
    for line in example_set:
        if line[0] == 'M':
            M_counter += 1
        else:
            B_counter += 1
    if M_counter == 0 or B_counter == 0:
        return True
    return False


def MajorityClass(example_set):      # True if |M| >= |B| , False if |M| < |B|
    M_counter = 0
    B_counter = 0
    for line in example_set:
        if line[0] == 'M':
            M_counter += 1
        else:
            B_counter += 1
    print('B_counter= ', B_counter, 'M_counter: ', M_counter)
    if M_counter >= B_counter:
        return True
    else:
        return False


def Classifier(data_frame, tree):
    if tree.leftNode is None and tree.rightNode is None:
        return tree.classification

    current_feature = tree.feature
    current_threshold = tree.threshold
    if data_frame[current_feature].iloc[0] < current_threshold:
        return Classifier(data_frame, tree.leftNode)
    else:
        return Classifier(data_frame, tree.rightNode)


def all_values_equal(example_set, feature_index):
    first = example_set[0][feature_index]
    for line in example_set:
        if line[feature_index] != first:
            return False
    return True


def experiment(m_param):  # experiment() , input: M parameter, output: accuracy, make sure train.csv in project dir

    # read csv file
    file = pd.read_csv('train.csv')

    # create DataFrame with pandas
    data_frame = pd.DataFrame(file)
    feature_set = [element for element in data_frame]

    accuracy_list = []
    sum = 0
    kf = KFold(n_splits=5, random_state=123456789, shuffle=True)
    splitted = kf.split(data_frame)

    for test in splitted:
        train_list = test[0]  # union of k-1 groups (train group)
        test_list = test[1]  # (test group)
        train_current_data_frame = data_frame.reindex(train_list)
        # print(train_current_data_frame)
        test_current_data_frame = data_frame.reindex(test_list)
        # print(test_current_data_frame)
        # print("\n\n")

        train_current_list = train_current_data_frame.to_numpy()

        tree = ID3(train_current_list, feature_set, True, m_param)  # learn on that training set
        accuracy, loss = calculate_accuracy_and_loss(test_current_data_frame, tree)  # test on the last piece
        accuracy_list.append(accuracy)
        print('accuracy: ', accuracy)
    for e in accuracy_list:
        sum += e
    average = sum / len(accuracy_list)
    print('* average * :  ', average)
    return average


def create_graph(lst):

    y_lst = []
    for e in lst:
        print("calculate for M = " + str(e))
        accuracy = experiment(e)
        y_lst.append(accuracy)

    plt.ylabel('Accuracy')
    plt.xlabel('M parameter')
    plt.plot(lst, y_lst)

    for i in range(len(lst)):
        f'{lst[i]:.5f}'
        #text = "(" + str(lst[i]) + ", " + str(y_lst[i]) + ")"
        text = "(" + str(lst[i]) + ", " + f'{y_lst[i]:.5f}' + ")"
        plt.text(lst[i], y_lst[i], s=text)

    plt.show()


def calculate_accuracy_and_loss(data_frame, tree):
    test_real_diagnosis = [diagnosis for diagnosis in data_frame["diagnosis"]]
    #print(test_real_diagnosis)
    correct_answer = 0
    wrong_answers = 0
    false_negative_counter = 0
    false_positive_counter = 0
    for i in range(len(test_real_diagnosis)):
        result = Classifier(data_frame.iloc[i:i + 1], tree)
        #result = better_classifier(data_frame.iloc[i:i + 1], tree)
        #print(result)
        real_diagnosis_bool = True if test_real_diagnosis[i] == 'M' else False
        #print('classifier result: ', result, 'Real result: ', real_diagnosis_bool)
        if result == real_diagnosis_bool:
            correct_answer += 1
        else:
            wrong_answers += 1
            if result is True and real_diagnosis_bool is False:  # this is False Positive situation
                false_positive_counter += 1

            if result is False and real_diagnosis_bool is True:  # False Negative situation
                print(i)
                false_negative_counter += 1
    print('wrong: ', wrong_answers)
    print('false_positive_counter: ', false_positive_counter, 'false_negative_counter:', false_negative_counter)
    accuracy = correct_answer / len(test_real_diagnosis)
    loss = ((0.1 * false_positive_counter) + (1 * false_negative_counter)) / len(test_real_diagnosis)
    return accuracy, loss


def run_system():  # run_system() no inputs, the output is accuracy, make sure train.csv and test.csv in the project dir

    # read csv files
    file = pd.read_csv('train.csv')
    file2 = pd.read_csv('test.csv')

    # create DataFrames with pandas
    data_frame = pd.DataFrame(file)
    data_frame_test = pd.DataFrame(file2)

    feature_set = [element for element in data_frame]
    example_set = data_frame.to_numpy()

    # todo step 1 : training
    tree = ID3(example_set, feature_set, True, 0)                   # main function

    # todo step 2 : test, then check accuracy and loss
    accuracy, loss = calculate_accuracy_and_loss(data_frame_test, tree)
    print(accuracy)

    # todo question 4.1
    print(loss)












