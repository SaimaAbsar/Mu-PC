# The mu-MMP main algorithm
# setting 1

# Saima Absar, CSCE, UARK
# May 25, 2021


import pandas as pd
import numpy as np

import mu_MMP_phase1 as phase1
import mu_MMP_phase2 as phase2


if __name__ == '__main__':

    n = 3
    G1 = phase2.Ginit
    print("Result of Phase-1:\n",G1)
    
    G2 = phase2.G2
    print("Result of Phase-2:\n",G2)
    
    f_r = np.array([G2.iloc[0,0], G2.iloc[0,1], G2.iloc[0,2], G2.iloc[1,0], G2.iloc[1,1], G2.iloc[1,2], G2.iloc[2,0], G2.iloc[2,1], G2.iloc[2,2]])
    a_r = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1])
    acc = np.sum(f_r == a_r)/len(a_r) * 100
    print('Percentage accuracy: ', acc)
    TP = (G2.iloc[0,0] + G2.iloc[0,1] + G2.iloc[2,1] + G2.iloc[2,2])
    FN = 4 - TP
    TP_FP = (G2.iloc[0,0]+ G2.iloc[0,1]+ G2.iloc[0,2]+ G2.iloc[1,0]+ G2.iloc[1,1]+ G2.iloc[1,2]+ G2.iloc[2,0]+ G2.iloc[2,1]+ G2.iloc[2,2])
    pre = TP / TP_FP
    rec = TP / (FN + TP)
    fm = (2 * pre * rec)/(pre + rec)
    #print('Percentage accuracy: ', acc)
    #print('Accuracy: ', acc)
    print('Precision: ', pre)
    print('Recall: ', rec)
    print('F-measure: ', fm)
    
    ## SHD
    
    from scipy.spatial import distance
    shd = distance.hamming(a_r, f_r)
    print('SHD: ', shd)
