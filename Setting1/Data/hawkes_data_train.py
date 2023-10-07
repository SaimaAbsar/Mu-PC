# Generates Hawkes time series for train set

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from random import expovariate, gauss, shuffle, randint

################################### for generating synthetic data
from scipy.stats import lognorm,gamma
from scipy.optimize import brentq

################################### for neural network modeling
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

np.random.seed(0)

###############################__x_z__############################

def generate_hawkes1():
    [T] = simulate_hawkes(1000,0.2,[0.6, 0.4],[1.0, 20.0])
    #score = - LL[32:].mean()
    return [T]
    
def generate_hawkes2():
    [T] = simulate_hawkes(1000,0.2,[0.6, 0.4],[1.0, 20.0])
    #score = - LL[32:].mean()
    return [T]
    
def simulate_hawkes(n,mu,alpha,beta):
    T = []
    LL = []
    
    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0
    
    while 1:
        l = mu + l_trg1 + l_trg2
        step = np.random.exponential()/l
        x = x + step
        
        l_trg_Int1 += l_trg1 * ( 1 - np.exp(-beta[0]*step) ) / beta[0]
        l_trg_Int2 += l_trg2 * ( 1 - np.exp(-beta[1]*step) ) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0]*step)
        l_trg2 *= np.exp(-beta[1]*step)
        l_next = mu + l_trg1 + l_trg2 
        
        if np.random.rand() < l_next/l: #accept
            T.append(x)
            LL.append( np.log(l_next) - l_trg_Int1 - l_trg_Int2 - mu_Int )
            l_trg1 += alpha[0]*beta[0]
            l_trg2 += alpha[1]*beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1
            
            if count == n:
                break
        
    #return [np.array(T),np.array(LL)]
    return [T]


###############################__y__############################
# Y points generated using x and z

def generate_hawkes3():
    [Tx] = generate_hawkes1()
    [Tz] = generate_hawkes2()
    [T] = simulate_hawkes_y(1000,0.5,[10, 100],[200, 200], Tx, Tz)

    return [T]



def simulate_hawkes_y(n,mu,alpha,beta, Tx, Tz):
    T = []
    LL = []
    step_y = 0

    x = 0
    l_trg1 = 0
    l_trg1_xz = 0
    l_trg2 = 0
    l_trg2_xz = 0

    count = 0
    l = mu
    index_x = 0
    index_z = 0
    index_a = 0
    index_b = 0
    
    while(1):
        
        l = mu + l_trg1_xz + l_trg2_xz 
        
        #print('l: ', l)
        step_y = np.random.exponential()/l
        #print('step_y: ', step_y)
        x = x + step_y

        l_trg1_xz *= np.exp(-beta[0]*step_y)

        l_trg2_xz *= np.exp(-beta[1]*step_y)

        j = index_x
        while j<len(Tx) and Tx[j]<x:

            l_trg1_xz += alpha[0]*beta[0] * (np.exp(-beta[0]*(x-Tx[j])))
            j += 1
            index_x = j
        
        k = index_z
        while k<len(Tz) and Tz[k]<x:

            l_trg1_xz += alpha[0]*beta[0] * (np.exp(-beta[0]*(x-Tz[k])))
            k += 1
            index_z = k
            
        a = index_a
        while a<len(Tx) and Tx[a]<x:

            l_trg2_xz += alpha[1]*beta[1] * (np.exp(-beta[1]*(x-Tx[a])))
            a += 1
            index_a = a
        
        b = index_b
        while b<len(Tz) and Tz[b]<x:

            l_trg2_xz += alpha[1]*beta[1] * (np.exp(-beta[1]*(x-Tz[b])))
            b += 1
            index_b = b

        l_next = mu + l_trg1_xz + l_trg2_xz
        
        if np.random.rand() < l_next/l: #accept
            T.append(x)

            l_trg1_xz += alpha[0]*beta[0]
            l_trg2_xz += alpha[1]*beta[1]

            count += 1
            
            if count == n:
                break 
    return [T]
#generate_hawkes3()




with open('time_alpha_1.txt', 'w') as output:

    score = []
    for i in range(1):
        [tmp] = generate_hawkes1()
        mstr = ''
        for ix in tmp:
            mstr += str(ix)+" "     
        output.write(mstr)
        output.write("\n")


with open('time_beta_1.txt', 'w') as output:

    score = []
    for i in range(1):
        [tmp] = generate_hawkes2()
        mstr = ''
        for ix in tmp:
            mstr += str(ix)+" "     
        output.write(mstr)
        output.write("\n")


with open('time_gamma_1.txt', 'w') as output:

    score = []
    for i in range(1):
        [tmp] = generate_hawkes3()
        mstr = ''
        for ix in tmp:
            mstr += str(ix)+" "     
        output.write(mstr)
        output.write("\n")


