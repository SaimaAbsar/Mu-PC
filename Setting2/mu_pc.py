
# The mu-PC algo including open-route check 
# setting 2

import subprocess
import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import chisquare
from scipy.stats import chi2

from check_open import open_route

class input:
    def __init__(self, name):
        self.name = name
        
    def path_distribution(self, path_line):
        paths = path_line.split(',')
     
        self.event = paths[0].strip()
        #self.time = paths[0].strip()
        self.event_test = paths[1].strip()
        #self.time_test = paths[1].strip() 
        

def likelihood_ratio_test(loss1, loss2, C):
    
    L1 = np.exp(-1 * loss1)
    L0 = np.exp(-1 * loss2)
    lambda_ = L0/L1;
    chi = -2 * np.log(lambda_)
    
    print('Chi: ', chi)
    
    
    print('lambda', lambda_)
    if chi >=C: return True
    elif loss1 < loss2: return True
    
    else: return False

def make_pair(n, C_, mode):
    pairs = []
    for i in range(n):
        if mode == 1:
            if (i == C_[0]): continue
        if mode == 2:
            if (i == C_[0] or i == C_[1]): continue
        else:
            if i == C_: continue
        for j in range(n):
            pairs.append([i,j])
    return(pairs)  


def evaluate_nn_0(alpha, beta):
    fo = open("datapath_upc.txt", "r")
    lines = fo.readlines()
    
    #print("line 1: ", lines[0])
    #print("line 2: ", lines[1])
    #print("line 3: ", lines[2])
    
    a = input('alpha')
    a.path_distribution(lines[alpha])
    b = input('beta')
    b.path_distribution(lines[beta])
    
    subprocess.call(['python','run_u_pc_m1i0.py', a.event, a.event_test, b.event, b.event_test, '--epochs', str(10)])

    subprocess.call(['python','run_u_pc_m2i0.py', a.event, a.event_test, b.event, b.event_test, '--epochs', str(8)])


def evaluate_nn_1(alpha, beta, C_):
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
  
    subprocess.call(['python','run_u_pc_m1i1.py', a.event, a.event_test, c_.event, c_.event_test, b.event, b.event_test, '--epochs', str(15)])

    subprocess.call(['python','run_u_pc_m2i1.py', a.event, a.event_test, c_.event, c_.event_test, b.event, b.event_test, '--epochs', str(12)])
    
    
def evaluate_nn_2(alpha, beta, C_):
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
  
    subprocess.call(['python','run_u_pc_m1i2.py', a.event, a.event_test, c_0.event, c_0.event_test, c_1.event, c_1.event_test, b.event, b.event_test, '--epochs', str(20)])

    subprocess.call(['python','run_u_pc_m2i2.py', a.event, a.event_test, c_0.event, c_0.event_test, c_1.event, c_1.event_test, b.event, b.event_test, '--epochs', str(18)])
 
    
def mu_pc_i0(G, n):
    C_ = []
    pairs = make_pair(n, C_, 0)
    for p in pairs:
        
        #############__Check u-connecting route_#################
        open_path = open_route(n, G, p, C_)
        
        #if G.iloc[p[0],p[1]] == 0: continue
        if not open_path: 
            print('No open path between %d and %d: '%(p[0],p[1]))
            continue
        else:
            print('open path between %d and %d: '%(p[0],p[1]), open_path)
            print('Evaluating pair %d and %d: '%(p[0],p[1]))
            evaluate_nn_0(p[0], p[1])
            #evaluate_nn(0, 2)
            
            with open('Loss_test_model1_i0.txt','r') as in_file:
                for x in in_file:
                    Loss1 = float(x)
                    
            with open('Loss_test_model2_i0.txt','r') as in_file:
                for x in in_file:
                    Loss2 = float(x)
            
            print('Model1: ',Loss1)
            print('Model0: ',Loss2)
           
            is_connected = likelihood_ratio_test(Loss1, Loss2, C)
            print('Edge from alpha %d to beta %d: '%(p[0],p[1]), is_connected)
            
            if not is_connected: G.iloc[p[0],p[1]] = 0
            else: G.iloc[p[0],p[1]] = 1
    print(G)
    #np.savetxt('G_i0.txt', G.values, delimiter="\t", header="X\tY\tZ")
    return(G)
    #print(is_connected)
    
def mu_pc_i1(G, n):
    
    for C_ in [[1], [0], [2]]: 
        #G = pd.DataFrame(matrix, columns=column_names, index=row_names)
        #G = G
        pairs = make_pair(n, C_, 1)
        print('Evaluation for C: ', C_)   
        for p in pairs:
        
            #############__Check u-connecting route_#################
            
            open_path = open_route(n, G, p, C_)
        
            #if G.iloc[p[0],p[1]] == 0: continue
            if not open_path: 
                print('No open path between %d and %d: '%(p[0],p[1]))
                continue
            else: 
                print('open path between %d and %d: '%(p[0],p[1]), open_path)
                print('Evaluating pair %d and %d: '%(p[0],p[1]))
                evaluate_nn_1(p[0], p[1], C_)
                #evaluate_nn(0, 2)
                
                with open('Loss_test_model1_i1.txt','r') as in_file:
                    for x in in_file:
                        Loss1 = float(x)
                        
                with open('Loss_test_model2_i1.txt','r') as in_file:
                    for x in in_file:
                        Loss2 = float(x)
                
                print('Model1: ',Loss1)
                print('Model0: ',Loss2)
                
                print('Checking Likelihood ratio') 
               
                is_connected = likelihood_ratio_test(Loss1, Loss2, C)
                print('Edge from alpha %d to beta %d: '%(p[0],p[1]), is_connected)
                
                if not is_connected: G.iloc[p[0],p[1]] = 0
                else: G.iloc[p[0],p[1]] = 1
                
        print('For C = %d: '%(C_[0]))
        print(G)
        #np.savetxt('G%d.txt' %C_[0], G.values, fmt='%d', delimiter="\t", header="X\tY\tZ")
    #print(is_connected)
    return(G)


def mu_pc_i2(G, n):
    
    for C_ in [[0,1], [0,2], [1,2]]:
        
    #for C_ in [[2,1]]: 
        #G = pd.DataFrame(matrix, columns=column_names, index=row_names)
        pairs = make_pair(n, C_, 2)
        print('Evaluation for C: ', C_)   
        for p in pairs:
            
            #############__Check u-connecting route_#################
        
            open_path = open_route(n, G, p, C_)
        
            #if G.iloc[p[0],p[1]] == 0: continue
            if not open_path: 
                print('No open path between %d and %d: '%(p[0],p[1]))
                continue
            else:
                print('open path between %d and %d: '%(p[0],p[1]), open_path)
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
                is_connected = likelihood_ratio_test(Loss1,Loss2,C)

                print('Edge from alpha %d to beta %d: '%(p[0],p[1]), is_connected)
                
                if not is_connected: G.iloc[p[0],p[1]] = 0
                else: G.iloc[p[0],p[1]] = 1
             
        print('For C: ', C_)
        print(G)
        #np.savetxt('G%d%d_i2.txt' %(C_[0],C_[1]), G.values, fmt='%d', delimiter="\t", header="X\tY\tZ")
    #print(is_connected)
    return(G)

    
if __name__ == '__main__':
    n = 3
    # assuming k = 1, C = 0.0001 level of significance = 0.01
    C = 0.0001 
    # a = alpha; b = beta; g = gamma
    # the graph represented as matrix format
    column_names = ['a', 'g', 'b']
    row_names    = ['a', 'g', 'b']
    matrix = np.ones((n,n), dtype=int)    
    G = pd.DataFrame(matrix, columns=column_names, index=row_names)

    G0 = mu_pc_i0(G, n)
    G1 = mu_pc_i1(G0, n)
    G2 = mu_pc_i2(G1, n)
    print('Final Result:\n', G2)
    f_r = np.array([G2.iloc[0,0], G2.iloc[0,1], G2.iloc[0,2], G2.iloc[1,0], G2.iloc[1,1], G2.iloc[1,2], G2.iloc[2,0], G2.iloc[2,1], G2.iloc[2,2]])
    a_r = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1])
    acc = np.sum(f_r == a_r)/len(a_r) #* 100
    #print('Percentage accuracy: ', acc)
    TP = (G2.iloc[0,0] + G2.iloc[0,1] + G2.iloc[2,1] + G2.iloc[2,2])
    
    #####---------------- for Exp-2:
    FN = 4 - TP
    TP_FP = (G2.iloc[0,0]+ G2.iloc[0,1]+ G2.iloc[0,2]+ G2.iloc[1,0]+ G2.iloc[1,1]+ G2.iloc[1,2]+ G2.iloc[2,0]+ G2.iloc[2,1]+ G2.iloc[2,2])
    pre = TP / TP_FP
    rec = TP / (FN+TP)
    fm = (2 * pre * rec)/(pre + rec)
    #print('Percentage accuracy: ', acc)
    print('Accuracy: ', acc)
    print('Precision: ', pre)
    print('Recall: ', rec)
    print('F-measure: ', fm)
