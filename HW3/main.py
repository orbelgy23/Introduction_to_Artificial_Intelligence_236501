
import ID3
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import CostSensitiveID3

from KNNForest import KNN_decision_tree
from KNNForest import KNN_decision_tree_experiment
import ImprovedKNNForest


if __name__ == "__main__":

    # todo Question 1 (ID3 check accuracy)
    # start = timer()
    # ID3.run_question_1()
    # end = timer()
    # print("total time: ", end - start)


    # todo Question 3 (K-fold cross validation)
    # lst1 = [1, 16, 45, 80, 120]

    # start = timer()
    # ID3.create_graph(lst1)
    # end = timer()
    # print("total time: ", end-start)

    # question 3.4
    # ID3.experiment(1)  # m = 1 is the best (Tested by experiments)


    # todo Question 4.1 (ID3 check loss)
    # ID3.run_question_4_1()


    # todo Question 4.3 (ID3 loss reduction by some manipulations)
    # start = timer()
    # CostSensitiveID3.run_question_4_3()
    # end = timer()
    # print("total time: ", end - start)


    # todo Question 6.1 (KNN check accuracy)
    # start = timer()
    # KNN_decision_tree_experiment(5)       # Given n, compute all the options for n and k - up to this n
    # KNN_decision_tree(4, 3)  # N = 4 , K = 3 the best parameters to choose (Tested by experiments)

    # end = timer()
    # print("total time: ", end - start)


    # todo Question 7.2 (KNN increasing accuracy by some manipulations)
    # for i in range(5):
    #ImprovedKNNForest.KNN_decision_tree(16, 15)
    #ImprovedKNNForest.KNN_decision_tree(16, 10)
    #ImprovedKNNForest.KNN_decision_tree(8, 5)  # got 1 sometimes
    #ImprovedKNNForest.KNN_decision_tree(7, 6)
    #ImprovedKNNForest.KNN_decision_tree(11, 8)
    ImprovedKNNForest.KNN_decision_tree_experiment(5)

    # ImprovedKNNForest.KNN_decision_tree(16, 10, 3)
    # ImprovedKNNForest.KNN_decision_tree(16, 10, 5)
    # ImprovedKNNForest.KNN_decision_tree(16, 15, 3)
    # ImprovedKNNForest.KNN_decision_tree(16, 15, 5)  # best - 1
    # ImprovedKNNForest.KNN_decision_tree(11, 8, 3)
    # ImprovedKNNForest.KNN_decision_tree(11, 8, 5)
    # ImprovedKNNForest.KNN_decision_tree(14, 2, 3)
    # ImprovedKNNForest.KNN_decision_tree(14, 2, 5)







