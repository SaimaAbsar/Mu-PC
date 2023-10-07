# The mu-MMP-Phase-1 
# setting 2

import subprocess
import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import chisquare
from scipy.stats.distributions import chi2


# To read the path directories to all the time series in the data
class input:
    def __init__(self, name):
        self.name = name
        
    def path_distribution(self, path_line):
        paths = path_line.split(',')
     
        self.event = paths[0].strip()
        #self.time = paths[0].strip()
        self.event_test = paths[1].strip()
        #self.time_test = paths[1].strip() 
        
        
        
# LRT score
def likelihood_ratio_test(loss1, loss2, C):
    
    print("In LRT test function")  
    
    L1 = np.exp(-1 * loss1)
    L0 = np.exp(-1 * loss2)
    lambda_ = L0/L1;
    chi = -2 * np.log(lambda_)
    
    print('Chi: ', chi)
    #print('Chi_event: ', chi_square_event)
    
    print('lambda', lambda_)
    
    '''
    if chi >=C: return True
    elif loss1 < loss2: return True
    #elif chi == nan: return True
    #if p_value<=0.01: return True
    #if loss1 < loss2: return True
    else: return False
    
    # degree of freedom = 1
    # LR = chi statistic
    # LR-score = Cumulative distribution function of a chi-squared random variable
    df = 1
    
    LR = 2 * (loss1 - loss2)
    LR_score = chi2.cdf(LR, df)
    p_value = 1 - LR_score
    
    print('LR: ', LR)
    print('LR_score: ', LR_score)
    print('p-value', p_value)
    '''
    
    LR_score = chi
    p_value = 1 - LR_score
    score = max([0,(LR_score-C)])
    
    return score
    
    
    
def evaluate_nn_0(alpha, beta, C_):
    
    print("Running model for i = 0; X = %d, Y = %d, U = " %(beta, alpha), C_)
    
    fo = open("datapath_upc.txt", "r")
    lines = fo.readlines()
    
    a = input('alpha')
    a.path_distribution(lines[alpha])
    b = input('beta')
    b.path_distribution(lines[beta])
    
    subprocess.call(['python','run_u_pc_m1i0.py', a.event, a.event_test, b.event, b.event_test, '--epochs', str(25)])

    subprocess.call(['python','run_u_pc_m2i0.py', a.event, a.event_test, b.event, b.event_test, '--epochs', str(20)])

    
    with open('Loss_test_model1_i0.txt','r') as in_file:
        for x in in_file:
            Loss1 = float(x)
            
    with open('Loss_test_model2_i0.txt','r') as in_file:
        for x in in_file:
            Loss2 = float(x)
            
    return(Loss1, Loss2)


def evaluate_nn_1(alpha, beta, C_):
    
    print("Running model for i = 1; X = %d, Y = %d, U = " %(beta, alpha), C_)
    
    fo = open("datapath_upc.txt", "r")
    lines = fo.readlines()
    
    #print("line 1: ", lines[0])
    #print("line 2: ", lines[1])
    #print("line 3: ", lines[2])
    
    a = input('alpha')
    a.path_distribution(lines[alpha])
    b = input('beta')
    b.path_distribution(lines[beta])
    c_ = input('C')
    c_.path_distribution(lines[C_[0]])
  
    subprocess.call(['python','run_u_pc_m1i1.py', a.event, a.event_test, c_.event, c_.event_test, b.event, b.event_test, '--epochs', str(25)])

    subprocess.call(['python','run_u_pc_m2i1.py', a.event, a.event_test, c_.event, c_.event_test, b.event, b.event_test, '--epochs', str(20)])
    
    with open('Loss_test_model1_i1.txt','r') as in_file:
        for x in in_file:
            Loss1 = float(x)
            
    with open('Loss_test_model2_i1.txt','r') as in_file:
        for x in in_file:
            Loss2 = float(x)
            
    return(Loss1, Loss2)
    
    
def evaluate_nn_2(alpha, beta, C_):
    
    print("Running model for i = 2; X = %d, Y = %d, U = " %(beta, alpha), C_)
    
    fo = open("datapath_upc.txt", "r")
    lines = fo.readlines()
    
    #print("line 1: ", lines[0])
    #print("line 2: ", lines[1])
    #print("line 3: ", lines[2])
    
    a = input('alpha')
    a.path_distribution(lines[alpha])
    b = input('beta')
    b.path_distribution(lines[beta])
    c_0 = input('C0')
    c_0.path_distribution(lines[C_[0]])
    c_1 = input('C1')
    c_1.path_distribution(lines[C_[1]])
  
    subprocess.call(['python','run_u_pc_m1i2.py', a.event, a.event_test, c_0.event, c_0.event_test, c_1.event, c_1.event_test, b.event, b.event_test, '--epochs', str(25)])

    subprocess.call(['python','run_u_pc_m2i2.py', a.event, a.event_test, c_0.event, c_0.event_test, c_1.event, c_1.event_test, b.event, b.event_test, '--epochs', str(20)])
 
    
    with open('Loss_test_model1_i2.txt','r') as in_file:
        for x in in_file:
            Loss1 = float(x)
            
    with open('Loss_test_model2_i2.txt','r') as in_file:
        for x in in_file:
            Loss2 = float(x)
            
    return(Loss1, Loss2)
    
    


def evaluate_nn_3(alpha, beta, C_):

    print("Running model for i = 3; X = %d, Y = %d, U = " %(beta, alpha), C_)

    fo = open(datapath, "r")
    lines = fo.readlines()
    
    #print("line 1: ", lines[0])
    #print("line 2: ", lines[1])
    #print("line 3: ", lines[2])
    
    a = input('alpha')
    a.path_distribution(lines[alpha])
    b = input('beta')
    b.path_distribution(lines[beta])
    c_0 = input('C0')
    c_0.path_distribution(lines[C_[0]])
    c_1 = input('C1')
    c_1.path_distribution(lines[C_[1]])
    c_2 = input('C2')
    c_2.path_distribution(lines[C_[2]])
  
    subprocess.call(['python','run_u_pc_m1i3.py', a.event, a.event_test, c_0.event, c_0.event_test, c_1.event, c_1.event_test, c_2.event, c_2.event_test, b.event, b.event_test, '--epochs', str(25)])

    subprocess.call(['python','run_u_pc_m2i3.py', a.event, a.event_test, c_0.event, c_0.event_test, c_1.event, c_1.event_test, c_2.event, c_2.event_test, b.event, b.event_test, '--epochs', str(20)]) 

    with open('Loss_test_model1_i3.txt','r') as in_file:
        for x in in_file:
            Loss1 = float(x)
            
    with open('Loss_test_model2_i3.txt','r') as in_file:
        for x in in_file:
            Loss2 = float(x)
            
    return(Loss1, Loss2)
    
    
    
    
# Phase-1 MMP algortihm

def mu_MMP(X, Y, U):
    
    print("In mu_MMP function")    
    
    if len(U) == 0: Loss1, Loss2 = evaluate_nn_0(Y, X, U)
    elif len(U) == 1: Loss1, Loss2 = evaluate_nn_1(Y, X, U)
    elif len(U) == 2: Loss1, Loss2 = evaluate_nn_2(Y, X, U)
    else: Loss1, Loss2 = evaluate_nn_3(Y, X, U)
    
    print('Model1: ',Loss1)
    print('Model0: ',Loss2)
   
    lrt_score = likelihood_ratio_test(Loss1, Loss2, C)
    
    return lrt_score
    
    
    
    
#if __name__ == '__main__':
n = 3

# the graph represented as matrix format
column_names = ['X', 'Y', 'Z']
row_names    = ['X', 'Y', 'Z']
matrix = np.zeros((n,n), dtype=int)    
G = pd.DataFrame(matrix, columns=column_names, index=row_names)

# assuming k = 1, C = 0.0001 level of significance = 0.01
# threshold = C
C = 0.0001

print("Running mu-MMP algoithm Phase-1")

for X in range(n):
    print("In loop for X = ", X)
    U = []
    
    while(1):
        
        score_list = []
        Y_list = []
        for Y in range(n):
            # Y is not in U
            if any(i==Y for i in U):
                print('Skipping Y = ', Y)
                continue
            
            scr = mu_MMP(X, Y, U)
            score_list.append(scr)
            Y_list.append(Y)
            
            print("Score for Y = %d: " %Y, scr)
        
        print("Score list: ", score_list)    
        
        # Repeat until X is independent of all the remaining nodes given U
        # i.e. score for all remaining nodes = 0
        if all(v == 0 for v in score_list): break
        
        max_value = max(score_list)
        max_index = score_list.index(max_value)
        U.append(Y_list[max_index])
        
        print('U: ', U)
     
    print("Step: X = %d, U = " %X, U)
    
    for i in U:
        G.iloc[i,X] = 1
        
#print(G)
    
    
