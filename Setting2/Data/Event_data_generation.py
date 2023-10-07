import numpy as np
from random import expovariate, gauss, shuffle, randint
import matplotlib.pyplot as plt

#np.random.seed(0)

def mult(interval, p1, p2):
    
    n = interval 
    y = []
    lly = 0
    m = np.random.multinomial(n, [p1, p2])

    for i in range(len(m)):
        y += [i+1]*m[i]
    shuffle(y)
    return y

def cond_prob(n, cond0, cond1):
    i0 = 0
    i1 = 0
    z = [randint(1, 2)]
    print(z)
    for i in range(n):
        t = z[i]
        if t==1:
            t1 = cond0[i0]
            i0+=1
        elif t == 2:
            t1 = cond1[i1]
            i1+=1
        z.append(t1)
        #print(z)
    return(z)

def extCaseInd(x, z, xcase, zcase):
    xz = np.ndarray([xcase+1, zcase+1, 0]).tolist()  # +1 because 0 is not a case
    #print(x)
    #print(z)
    for i in range(len(x)):
        #print (i)
        xz[x[i]][z[i]].append(i)    # extract the index of each case
    return xz

def cond_prob_y(n, x, z):
    i0 = 0
    i1 = 0
    i2 = 0
    i3 = 0
    
    '''
    xz = extCaseInd(x,z,3,3)
    
    for i in range(1,4):
        for j in range(1,4):
            print("%d %d: %s" % (i,j,str(xz[i][j])))
    '''
    y_cond0 = mult(n, 0.9, 0.1)
    y_cond1 = mult(n, 0.7, 0.3) 
    y_cond2 = mult(n, 0.3, 0.7)  
    y_cond3 = mult(n, 0.1, 0.9)
    
    
    y = [randint(1, 2)]
    print(y)
    for i in range(n):
        t = [x[i], z[i]]
        #t = [x[i]]
        if t==[1, 1]:
            t1 = y_cond0[i0]
            i0+=1
        elif t == [1,2]:
            t1 = y_cond1[i1]
            i1+=1
        elif t == [2,1]:
            t1 = y_cond2[i2]
            i2+=1
        elif t==[2,2]:
            t1 = y_cond3[i3]
            i3+=1
        
        y.append(t1)
        #print(z)
    return(y)    
        
if __name__ == '__main__':
    n = 1000
    x = [randint(1, 2)]
    z = [randint(1, 2)]
    x_test = [randint(1, 2)]
    z_test = [randint(1, 2)]
    x_cond0 = mult(n, 0.5, 0.5)
    
    #x_cond0 = mult(n, 0.75, 0.25)
    #x_cond1 = mult(n, 0.1, 0.9) 
    #x = cond_prob(n, x_cond0, x_cond1)
    #x_cond2 = mult(n, 0.4, 0.3, 0.3)       
    x = x + x_cond0
    
    z_cond0 = mult(n, 0.5, 0.5)
    #z_cond0 = mult(n, 0.2, 0.8)
    #z_cond1 = mult(n, 0.95, 0.05) 
    #z_cond2 = mult(n, 0.5, 0.2, 0.3)       
    #z = cond_prob(n, z_cond0, z_cond1)	
    z = z + z_cond0
             
    y = cond_prob_y(n, x, z)	
    
    
    n = 400
    
    x_cond0 = mult(n, 0.5, 0.5)
    #x_cond0 = mult(n, 0.75, 0.25)
    #x_cond1 = mult(n, 0.1, 0.9) 
    #x_cond2 = mult(n, 0.4, 0.3, 0.3)
    #x_test = cond_prob(n, x_cond0, x_cond1)       
    x_test = x_test + x_cond0 # cond_prob(n, x_cond0, x_cond1)	
    
    z_cond0 = mult(n, 0.5, 0.5)
    #z_cond0 = mult(n, 0.2, 0.8)
    #z_cond1 = mult(n, 0.95, 0.05) 
    #z_cond2 = mult(n, 0.5, 0.2, 0.3)       
    #z_test = cond_prob(n, z_cond0, z_cond1)	
    z_test = z_test + z_cond0
             
    y_test = cond_prob_y(n, x_test, z_test)
    
#print(x)
#print(z)
#print(y)	

#######--------write_data_in_file---------#######
with open('alpha10.txt', 'w') as output:
    for i in range(1):
        mstr = ''
        for ix in x:
            mstr += str(ix)+" "     
        output.write(mstr)
        output.write("\n")

with open('beta10.txt', 'w') as output:
    for i in range(1):
        mstr = ''
        for ix in z:
            mstr += str(ix)+" "     
        output.write(mstr)
        output.write("\n")
       

with open('gamma10.txt', 'w') as output:
    mstr = ''
    for ix in y:
        mstr += str(ix)+" "     
    output.write(mstr)
    output.write("\n")
    
    
with open('alpha_test10.txt', 'w') as output:
    for i in range(1):
        mstr = ''
        for ix in x_test:
            mstr += str(ix)+" "     
        output.write(mstr)
        output.write("\n")

with open('beta_test10.txt', 'w') as output:
    for i in range(1):
        mstr = ''
        for ix in z_test:
            mstr += str(ix)+" "     
        output.write(mstr)
        output.write("\n")
       

with open('gamma_test10.txt', 'w') as output:
    mstr = ''
    for ix in y_test:
        mstr += str(ix)+" "     
    output.write(mstr)
    output.write("\n")

