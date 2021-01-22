import pandas as pd
from ID3 import ID3


def calculate_accuracy_and_loss(data_frame, tree):
    test_real_diagnosis = [diagnosis for diagnosis in data_frame["diagnosis"]]
    correct_answer = 0
    wrong_answers = 0
    false_negative_counter = 0
    false_positive_counter = 0

    for i in range(len(test_real_diagnosis)):
        result = better_classifier(data_frame.iloc[i:i + 1], tree)

        real_diagnosis_bool = True if test_real_diagnosis[i] == 'M' else False

        if result == real_diagnosis_bool:
            correct_answer += 1
        else:
            wrong_answers += 1
            if result is True and real_diagnosis_bool is False:  # this is False Positive situation
                false_positive_counter += 1

            if result is False and real_diagnosis_bool is True:  # False Negative situation
                false_negative_counter += 1

    # print('false_positive_counter: ', false_positive_counter, 'false_negative_counter:', false_negative_counter)
    accuracy = correct_answer / len(test_real_diagnosis)
    loss = ((0.1 * false_positive_counter) + (1 * false_negative_counter)) / len(test_real_diagnosis)
    return accuracy, loss


def better_classifier(data_frame, tree):
    if tree.leftNode is None and tree.rightNode is None:
        return tree.classification

    current_feature = tree.feature
    current_threshold = tree.threshold

    # improvement for leafs
    if abs(data_frame[current_feature].iloc[0] / current_threshold - 1) < 0.125:       # ~0.1-0.15 improvement
        if data_frame[current_feature].iloc[0] >= current_threshold and tree.rightNode.classification is False and tree.leftNode.classification is True:
            return better_classifier(data_frame, tree.leftNode)
        elif data_frame[current_feature].iloc[0] < current_threshold and tree.leftNode.classification is False and tree.rightNode.classification is True:
            return better_classifier(data_frame, tree.rightNode)
        else:                                                     # same as before
            if data_frame[current_feature].iloc[0] < current_threshold:
                return better_classifier(data_frame, tree.leftNode)
            else:
                return better_classifier(data_frame, tree.rightNode)
    else:                                                        # same as before
        if data_frame[current_feature].iloc[0] < current_threshold:
            return better_classifier(data_frame, tree.leftNode)
        else:
            return better_classifier(data_frame, tree.rightNode)


def run_question_4_3():  # no inputs, the output is loss, make sure train.csv and test.csv in the project dir

    # read csv files
    file = pd.read_csv('train.csv')
    file2 = pd.read_csv('test.csv')

    # create DataFrames with pandas
    data_frame = pd.DataFrame(file)
    data_frame_test = pd.DataFrame(file2)

    feature_set = [element for element in data_frame]
    example_set = data_frame.to_numpy()

    # todo step 1 : training
    tree = ID3(example_set=example_set, feature_set=feature_set, classification=True, m_param=5)  # m = 5 best for reducing the loss

    # todo step 2 : test, then check accuracy and loss
    accuracy, loss = calculate_accuracy_and_loss(data_frame_test, tree)  # function uses better_classifier()
    print(loss)
