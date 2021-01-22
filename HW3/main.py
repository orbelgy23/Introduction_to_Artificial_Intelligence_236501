
import ID3
import matplotlib.pyplot as plt
import CostSensitiveID3
from KNNForest import KNN_decision_tree
from KNNForest import KNN_decision_tree_experiment
import ImprovedKNNForest


if __name__ == "__main__":

    # todo Question 1 (ID3 check accuracy)
    # ID3.run_question_1()

    # todo Question 3 (K-fold cross validation)
    # lst1 = [1, 16, 45, 80, 120]
    # ID3.create_graph(lst1)

    # question 3.4
    # ID3.experiment(1)  # m = 1 is the best (Tested by experiments)

    # todo Question 4.1 (ID3 check loss)
    # ID3.run_question_4_1()

    # todo Question 4.3 (ID3 loss reduction by some manipulations)
    # CostSensitiveID3.run_question_4_3()

    # todo Question 6.1 (KNN check accuracy)
    # KNN_decision_tree_experiment(5)       # Given n, compute all the options for n and k - up to this n
    # KNN_decision_tree(N_param=4, K_param=3)  # N = 4 , K = 3 the best parameters to choose (Tested by experiments)

    # todo Question 7.2 (KNN increasing accuracy by some manipulations)
    # ImprovedKNNForest.KNN_decision_tree_experiment(5)     # experiment 1

    # for i in range(10):                                   # ImprovedKNN vs. KNN results
    #     ImprovedKNNForest.KNN_decision_tree(4, 3, 5)
    #     KNN_decision_tree(4, 3)
    #     print('----------')

    ImprovedKNNForest.KNN_decision_tree_random_experiment()     # for question 7.2
