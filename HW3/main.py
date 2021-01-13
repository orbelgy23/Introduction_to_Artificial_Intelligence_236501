
from ID3 import *
import matplotlib.pyplot as plt
from timeit import default_timer as timer


if __name__ == "__main__":

    # todo Question 1
    # start = timer()
    # run_system()
    # end = timer()
    # print("total time: ", end - start)


    # todo Question 3 (K-fold cross validation)
    lst1 = [1, 2, 3, 5, 8, 16, 30, 50, 80, 120]
    #lst2 = [10, 75, 150, 225, 300]
    #lst3 = [100]

    # experiment(0)

    start = timer()
    create_graph(lst1)
    end = timer()
    print("total time: ", end-start)













