
import ID3
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import CostSensitiveID3

from KNNForest import KNN_decision_tree
from KNNForest import KNN_decision_tree_experiment


if __name__ == "__main__":

    # todo Question 1 (ID3 check accuracy)
    # start = timer()
    ID3.run_question_1()
    # end = timer()
    # print("total time: ", end - start)


    # todo Question 3 (K-fold cross validation)
    # lst1 = [1, 2, 3, 5, 8, 16, 30, 50, 80, 120]

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
    # KNN_decision_tree_experiment(4)       # Given n, compute all the options for n and k - up to this n
    # KNN_decision_tree(4, 3)               # N = 4 , K = 3 the best parameters to choose (Tested by experiments)
    # end = timer()
    # print("total time: ", end - start)


    # todo Question 7.2 (KNN increasing accuracy by some manipulations)







