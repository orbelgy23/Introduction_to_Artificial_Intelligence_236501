
import ID3
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import CostSensitiveID3

from KNNForest import KNN_decision_tree

if __name__ == "__main__":

    # todo Question 1 + 4.1
    # start = timer()
    # ID3.run_system()
    # end = timer()
    # print("total time: ", end - start)


    # todo Question 3 (K-fold cross validation)
    # lst1 = [1, 2, 3, 5, 8, 16, 30, 50, 80, 120]
    # lst2 = [10, 75, 150, 225, 300]

    # ID3.experiment(0)

    # start = timer()
    # ID3.create_graph(lst1)
    # end = timer()
    # print("total time: ", end-start)


    # todo Question 4.2
    start = timer()
    CostSensitiveID3.run_system()
    end = timer()
    print("total time: ", end - start)



    # todo Question 6
    # start = timer()
    # KNN_decision_tree(4, 3)
    # end = timer()
    # print("total time: ", end - start)






