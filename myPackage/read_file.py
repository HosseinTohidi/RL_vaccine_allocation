# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:56:18 2020

@author: atohidi
"""
import numpy as np
def read_file(fileName, sum_of_all_groups = 1000, num_age_group= 20):
    print("--------------------------------------------------------------")        
    print('Reading file ...')
    file = open(fileName)
    counter = 0
    contact_rates = []
    for row in file:
        if row in ["\n", " "]:
            pass
        else:
            if counter == 0:
                groups_num = int(row)
                counter+=1             
            elif counter == 1:
                totalPopulation = list(map(float, row.split('\t')[:-1])) 
                totalPopulation = [totalPopulation[i]*sum_of_all_groups for i in range(groups_num)]
                totalPopulation = list(map(int,totalPopulation))
                counter+=1 
            elif counter == 2:
                initialinfeactions = list(map(float, row.split())) 
                counter+=1 
            elif counter == 3:
                contact_rates.append(list(map(float, row.split())))
                if len(contact_rates) == groups_num:
                    counter+=1
            elif counter == 4:
                vaccineEfficacy = float(row[:-1])
                counter+=1
            elif counter in {5,6,7}:
                counter+=1
                if counter == 7:
                    print('vaccination, pathogenecity, and prob of qurantine informations are ignored')
            elif counter == 8:
                np.random.seed(12345)
                omega = 0.11 + np.round(np.random.random(size = [groups_num])/10,2)
                print('omega is ignored and replaced by a random vector of range[0,0.25] with seed 12345, rounded by two digits')
                counter += 1
            elif counter == 9:
                gamma = list(map(float, row.split()))
                counter+=1
            elif counter == 10:
                H = list(map(float, row.split()))
                counter+=1
            elif counter == 11:
                RS = list(map(float, row.split()))
                counter+=1            
    print('File is read successfully')
    print("--------------------------------------------------------------") 
    # only choose up to num_age_group
    groups_num = num_age_group
    totalPopulation = totalPopulation[:groups_num]
    contact_rates = np.array(contact_rates)[:groups_num,:groups_num]
    H = H[:groups_num]
    RS = RS[:groups_num]       
    initialinfeactions = initialinfeactions[:groups_num]
    omega = omega[:groups_num]
    gamma = gamma[:groups_num]

    return groups_num, totalPopulation, initialinfeactions, contact_rates, vaccineEfficacy, omega, gamma, H, RS