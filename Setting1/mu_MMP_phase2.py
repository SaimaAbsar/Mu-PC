# The mu-MMP-Phase-2 
# setting 1

import subprocess
import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import chisquare
from scipy.stats.distributions import chi2
#import mu_MMP_phase1 as phase1
from check_open import open_route

datapath = "datapath_upc.txt"

# To read the path directories to all the time series in the data
class input:
    def __init__(self, name):
        self.name = name
        
    def path_distribution(self, path_line):
        paths = path_line.split(',')
     
        self.event = paths[0].strip()
        self.time = paths[1].strip()
        self.event_test = paths[2].strip()
        self.time_test = paths[3].strip() 
        
        
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
    if chi >=C: return True
    elif loss1 < loss2: return True
    #elif chi == nan: return True
    #if p_value<=0.01: return True
    #if loss1 < loss2: return True
    else: return False
    
    '''
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
    
    score = max([0,(LR_score-C)])
    
    return score
    '''
    

# Function which returns subset or r length from n
from itertools import combinations
  
def rSubset(r):
  
    arr = list(range(n))
    
    # return list of all subsets of length r
    return list(combinations(arr, r))


# Function to check for existence of direct edge
def direct_edge(G,i,C_):
    for j in C_:
        if (G.iloc[i,j] == 1 or G.iloc[j,i] == 1):
            r = True
        else: r = False  
    return r  
        

# make pair(alpha, beta) for C=null
def make_pair(n, C_):
    pairs = []
    for i in range(n):
        for j in range(n):
            pairs.append([i,j])
    return(pairs)  


# make pair(alpha, beta), alpha not in C & alpha adj. to C
def make_pair_adj(G, n, C_):
    pairs = []
    for i in range(n):
        if any(y==i for y in C_): continue
        if direct_edge(G, i, C_):
            for j in range(n):
                pairs.append([i,j])
    return(pairs)  


#########################################################
### RMTPP model calls

def evaluate_nn_0(alpha, beta):
    fo = open(datapath, "r")
    lines = fo.readlines()
    
    #print("line 1: ", lines[0])
    #print("line 2: ", lines[1])
    #print("line 3: ", lines[2])
    
    a = input('alpha')
    a.path_distribution(lines[alpha])
    b = input('beta')
    b.path_distribution(lines[beta])
    
    subprocess.call(['python','run_u_pc_m1i0.py', a.event, a.time, a.event_test, a.time_test, b.event, b.time, b.event_test, b.time_test, '--epochs', str(1)])

    subprocess.call(['python','run_u_pc_m2i0.py', a.event, a.time, a.event_test, a.time_test, b.event, b.time, b.event_test, b.time_test, '--epochs', str(1)])


def evaluate_nn_1(alpha, beta, C_):
    fo = open(datapath, "r")
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
  
    subprocess.call(['python','run_u_pc_m1i1.py', a.event, a.time, a.event_test, a.time_test, c_.event, c_.time, c_.event_test, c_.time_test, b.event, b.time, b.event_test, b.time_test, '--epochs', str(1)])

    subprocess.call(['python','run_u_pc_m2i1.py', a.event, a.time, a.event_test, a.time_test, c_.event, c_.time, c_.event_test, c_.time_test, b.event, b.time, b.event_test, b.time_test, '--epochs', str(1)])
    
    
    
def evaluate_nn_2(alpha, beta, C_):
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
  
    subprocess.call(['python','run_u_pc_m1i2.py', a.event, a.time, a.event_test, a.time_test, c_0.event, c_0.time, c_0.event_test, c_0.time_test, c_1.event, c_1.time, c_1.event_test, c_1.time_test, b.event, b.time, b.event_test, b.time_test, '--epochs', str(1)])

    subprocess.call(['python','run_u_pc_m2i2.py', a.event, a.time, a.event_test, a.time_test, c_0.event, c_0.time, c_0.event_test, c_0.time_test, c_1.event, c_1.time, c_1.event_test, c_1.time_test, b.event, b.time, b.event_test, b.time_test, '--epochs', str(1)])
 
    
def mu_pc_i0(G, n):
    C_ = []
    pairs = make_pair(n, C_)
    for p in pairs:
    
        #############__Check u-connecting route_#################
        #open_path = open_route(n, G, p, C_)
    
        if G.iloc[p[0],p[1]] == 0: continue
        #if not open_path: 
        #    print('No open path between %d and %d: '%(p[0],p[1]))
        #    continue
        else:
            #print('open path between %d and %d: '%(p[0],p[1]), open_path)
            print('Evaluating pair %d and %d: '%(p[0],p[1]))
            evaluate_nn_0(p[0], p[1])
            #evaluate_nn(0, 2)
            
            x = open('Loss_test_model1_i0.txt','r').read()
            if x == '': 
                Loss1 = None
                continue
            else: Loss1 = float(x)
            x = open('Loss_test_model2_i0.txt','r').read()
            if x == '': 
                Loss2 = None
                continue
            else: Loss2 = float(x)
            
            print('Model1: ',Loss1)
            print('Model0: ',Loss2)
            
            if Loss1 != 0:
                is_connected = likelihood_ratio_test(Loss1, Loss2, C)
            else: is_connected = True    
            
            print('Edge from alpha %d to beta %d: '%(p[0],p[1]), is_connected)
            
            if not is_connected: G.iloc[p[0],p[1]] = 0
            #else: G.iloc[p[0],p[1]] = 1
        
        #else: continue
            
    print(G)
    #np.savetxt('G_i0.txt', G.values, fmt='%d', delimiter="\t", header="TMP\tFRS\tCH4\tCO2\tGL0\tPRE\tVAP\tCLD\tWET")
    return(G)
    #print(is_connected)
    
def mu_pc_i1(G, n):
    
    mode = 1
    for C_ in rSubset(mode):  
        
        pairs = make_pair_adj(G, n, C_)
        print('Evaluation for C: ', C_)   
        for p in pairs:
            
            #############__Check u-connecting route_#################
            
            #open_path = open_route(n, G, p, C_)
        
            if G.iloc[p[0],p[1]] == 0: continue
            #if not open_path: 
            #    print('No open path between %d and %d: '%(p[0],p[1]))
            #    continue
            else:
                #print('open path: ', open_path)
                print('Evaluating pair %d and %d: '%(p[0],p[1]))
                evaluate_nn_1(p[0], p[1], C_)
                #evaluate_nn(0, 2)
                
                x = open('Loss_test_model1_i1.txt','r').read()
                if x == '': 
                    Loss1 = None
                    continue
                else: Loss1 = float(x)
                x = open('Loss_test_model2_i1.txt','r').read()
                if x == '': 
                    Loss2 = None
                    continue
                else: Loss2 = float(x)
                
                
                
                print('Model1: ',Loss1)
                print('Model0: ',Loss2)
                
                print('Checking Likelihood ratio') 
               
                
                if Loss1 == 0: is_connected = True
                
                else: is_connected = likelihood_ratio_test(Loss1, Loss2, C)
                print('Edge from alpha %d to beta %d: '%(p[0],p[1]), is_connected)
                
                if not is_connected: G.iloc[p[0],p[1]] = 0
                #else: G.iloc[p[0],p[1]] = 1
             
                
        print('For C = %d: '%(C_[0]))
        print(G)
        #np.savetxt('G%d_i1.txt' %(C_[0]), G.values, fmt='%d', delimiter="\t", header="TMP\tFRS\tCH4\tCO2\tGL0\tPRE\tVAP\tCLD\tWET")
    #print(is_connected)
    return(G)


def mu_pc_i2(G, n):
    
    mode = 2
    for C_ in rSubset(mode):
        
    #for C_ in [[2,1]]: 
        #G = pd.DataFrame(matrix, columns=column_names, index=row_names)
        pairs = make_pair_adj(G, n, C_)
        print('Evaluation for C: ', C_)   
        for p in pairs:
        
            #############__Check u-connecting route_#################
        
            #open_path = open_route(n, G, p, C_)
        
            if G.iloc[p[0],p[1]] == 0: continue
            #if not open_path: 
            #    print('No open path between %d and %d: '%(p[0],p[1]))
            #    continue
            else:
                #print('open path between %d and %d: '%(p[0],p[1]), open_path)
                print('Evaluating pair %d and %d: '%(p[0],p[1]))
                evaluate_nn_2(p[0], p[1], C_)
                #evaluate_nn(0, 2)
                
                
                x = open('Loss_test_model1_i2.txt','r').read()
                if x == '': 
                    Loss1 = None
                    continue
                else: Loss1 = float(x)
                x = open('Loss_test_model2_i2.txt','r').read()
                if x == '': 
                    Loss2 = None
                    continue
                else: Loss2 = float(x)
                
                
                print('Checking Likelihood ratio') 
                
                
                print('Model1: ',Loss1)
                print('Model0: ',Loss2)
                print('Checking Likelihood ratio') 
                if Loss1 == 0: is_connected = True
                else: is_connected = likelihood_ratio_test(Loss1,Loss2,C)

                print('Edge from alpha %d to beta %d: '%(p[0],p[1]), is_connected)
                
                if not is_connected: G.iloc[p[0],p[1]] = 0
                #else: G.iloc[p[0],p[1]] = 1
             
        print('For C: ', C_)
        print(G)
        #np.savetxt('G%d%d_i2.txt' %(C_[0],C_[1]), G.values, fmt='%d', delimiter="\t", header="TMP\tFRS\tCH4\tCO2\tGL0\tPRE\tVAP\tCLD\tWET")
    #print(is_connected)
    return(G)



# Importing The graph from phase-1
n = 3

# assuming k = 1, C = 0.0001 level of significance = 0.01
# threshold = C
C = 0.0001

'''
column_names = ['X', 'Y', 'Z']
row_names    = ['X', 'Y', 'Z']
matrix = np.loadtxt('G_phase-1_Set1.txt', dtype=int, usecols=range(3))
#matrix = np.ones((n,n), dtype=int)    
G_phase1 = pd.DataFrame(matrix, columns=column_names, index=row_names)

'''
G_phase1 = phase1.G
Ginit = phase1.G.copy(deep=True)
#print(G_phase1)

print("Result of Phase-1:\n",G_phase1)

# Phase-2 Calculations

print("Running mu-MMP algorithm Phase-2")

G0 = mu_pc_i0(G_phase1, n)
G1 = mu_pc_i1(G0, n)
G2 = mu_pc_i2(G1, n)

# Final result = G2
#Ginit = phase1.G
#print("Result of Phase-1:\n",Ginit)
#print("Result of Phase-2:\n",G2)
