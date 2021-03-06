# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 23:25:35 2020

@author: atohidi
"""

class MyClass():
    def __init__(self, input):
        self.input = input
        self.result = int

    def my_process(self, multiply_by, add_to):
        self.result = self.input * multiply_by
        self._my_sub_process(add_to)
        return self.result

    def _my_sub_process(self, add_to):
        self.result += add_to

import multiprocessing as mp
NUM_CORE = 4  # set to the number of cores you want to use

def worker(arg):
    obj, m, a = arg
    return obj.my_process(m, a)

if __name__ == "__main__":
    list_of_numbers = range(0, 5)
    list_of_objects = [MyClass(i) for i in list_of_numbers]

    pool = mp.Pool(NUM_CORE)
    list_of_results = pool.map(worker, ((obj, 100, 1) for obj in list_of_objects))
    pool.close()
    pool.join()

    print(list_of_numbers)
    print(list_of_results)