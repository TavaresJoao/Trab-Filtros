# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:25:02 2019

@author: joao_
"""

#%%
#import ctrl
import math
import scipy
import control
import numpy as np
from control.matlab import *
from control import TransferFunction
from scipy import signal
from math import log
import matplotlib.pyplot as plt

#%% Entradas
f1 = 13e3
f2 = 28e3

w_1 = 2 * math.pi * f1
w_2 = 2 * math.pi * f2

Gdb1 = -0.5
Gdb2 = -18

#%%
G1 = 10 ** (Gdb1/20)

G2 = 10 ** (Gdb2/20)

#%%
aux_1 = 1/(G1 ** 2) - 1
aux_2 = 1/(G2 ** 2) - 1

#%%
n = 1/2 * ( log(aux_1/aux_2) / log(f1/f2) )
n =  math.ceil(n)

w_0 = ( (w_1**(2*n))/aux_1 ) ** (1/(2*n))
f0 = w_0 / (2*math.pi)

#%%
#s = TransferFunction.s

#aux = - s**2 / (4 * math.pi**2 * f0**2)
#aux = aux**n
#%%
f = np.arange(1e0, 1e5, 10)

aux = f / f0
aux = aux**(2*n)
H = 1 / ( ( 1+aux )**(1/2) )

plt.semilogx(f, H)
plt.axis([1e3, 1e5, H.min()-0.1, H.max()+0.1])
plt.grid()
plt.show()

#%%
H_db = 20*np.log10(H)

plt.semilogx(f, H_db)
plt.axis([1e3, 1e5, H_db.min()-1, H_db.max()+1])
plt.grid()
plt.show()

#%%
fig = plt.figure()

ax = fig.add_subplot(111)

plt.plot(f, H, '-b')

plt.axvline(x=f1, color='red', label='f1')
plt.axvline(x=f2, color='red', label='f2')

#plt.xlim(1e5, 1e7)
#plt.ylim(H.min()-0.1, H.max()+0.1)
plt.axis([1e3, 1e5, H.min()-0.1, H.max()+0.1])
plt.grid()
plt.legend()
plt.show()

#%%
pk = np.ones(n,dtype = 'complex') 

for i in range(1,n+1):
    pk[i-1] = complex((w_0*math.cos(((n+2*(i)-1)*math.pi)/(2*n))),((w_0*math.sin(((n+2*(i)-1)*math.pi)/(2*n)))))
pk[math.floor(n/2)] = 0;

H = []

denominadores = np.ones((math.floor(n/2),3),dtype='complex')
numeradores = np.ones((math.floor(n/2), 1), dtype='complex')
for i in range (0,math.floor(n/2)):
    denominadores[i] = np.array([1, -pk[i]-pk[n-1-i], pk[i]*pk[n-1-i]])
    numeradores[i] = np.array([pk[i]*pk[n-1-i]])
    
    sys = tf(numeradores[i], denominadores[i])
    H.append(sys)

C3 = 1.3e-9
R3 = 1 / ( C3*2*math.pi*w_0 )