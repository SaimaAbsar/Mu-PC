# Generate marker points with respect to the Hawkes timings

# x refers to alpha, z refers to beta and y refers to gamma

import numpy as np
from random import expovariate, gauss, shuffle, randint


import matplotlib.pyplot as plt

#np.random.seed(10)

def event_data(interval, p1, p2):
    
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
    for i in range(n-1):
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

###########------generate_y---####################


with open('time_alpha_1.txt','r') as in_file:
    for x in in_file:
        Tx = [float(y) for y in x.strip().split()] 

with open('time_beta_1.txt','r') as in_file:
    for x in in_file:
        Tz = [float(y) for y in x.strip().split()] 

with open('time_gamma_1.txt','r') as in_file:
    for x in in_file:
        Ty = [float(y) for y in x.strip().split()] 

n = 1000

x_cond0 = event_data(n, 0.1, 0.9)
x_cond1 = event_data(n, 0.3, 0.7) 
event_x = cond_prob(n, x_cond0, x_cond1)
#x_cond2 = mult(n, 0.4, 0.3, 0.3)       
#x = x + x_cond0

#z_cond0 = mult(n, 0.4, 0.6)
z_cond0 = event_data(n, 0.85, 0.15)
z_cond1 = event_data(n, 0.35, 0.65) 
#z_cond2 = mult(n, 0.5, 0.2, 0.3)       
event_z = cond_prob(n, z_cond0, z_cond1)	

#event_x = event_data(1000, 0.5, 0.5)
#event_z = event_data(1000, 0.4, 0.6)  

event_y11 = event_data(1000, 0.9, 0.1)
event_y12 = event_data(1000, 0.7, 0.3)    
event_y21 = event_data(1000, 0.3, 0.7)
event_y22 = event_data(1000, 0.1, 0.9)  

index_x = 0
index_z = 0
index_y = 0
 

event_y = []
for i in range(len(Ty)):
    j = index_x
    while j<len(Tx) and Tx[j]<Ty[i]:
        #np.append(diffx,(Ty[i]-Tx[j]))
        index_x = j
        j += 1
        
    k = index_z
    while k<len(Tz) and Tz[k]<Ty[i]:
        #np.append(diffz,(Ty[i]-Tz[k]))
        index_z = k
        k += 1

    if event_x[index_x] == 1:
        if event_z[index_z] == 1:
            event_y.append(event_y11[i])
        else:
            event_y.append(event_y12[i])
    elif event_x[index_x] == 2:
        if event_z[index_z] == 1:
            event_y.append(event_y21[i])
        else:
            event_y.append(event_y22[i])

print(event_x[0])
print(event_z[0])
print(event_y[0])
print(len(event_x))


#######--------write_data_in_file---------#######
with open('event_alpha_1.txt', 'w') as output:
    for i in range(1):
        mstr = ''
        for ix in event_x:
            mstr += str(ix)+" "     
        output.write(mstr)
        output.write("\n")
     
with open('event_beta_1.txt', 'w') as output:
    for i in range(1):
        mstr = ''
        for ix in event_z:
            mstr += str(ix)+" "     
        output.write(mstr)
        output.write("\n")      
      
with open('event_gamma_1.txt', 'w') as output:
    mstr = ''
    for ix in event_y:
        mstr += str(ix)+" "     
    output.write(mstr)
    output.write("\n")
    

