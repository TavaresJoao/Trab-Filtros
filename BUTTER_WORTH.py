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
f1 = 1e6
f2 = 2e6

w_1 = 2 * math.pi * f1
w_2 = 2 * math.pi * f2

Gdb1 = -0.45
Gdb2 = -9

#%%
G1 = 10 ** (Gdb1/20)

G2 = 10 ** (Gdb2/20)

#%%
aux_1 = 1/(G1 ** 2) - 1
aux_2 = 1/(G2 ** 2) - 1

#%%
n = 1/2 * ( log(aux_1/aux_2) / log(f1/f2) )
n =  round(n)

w_0 = ( (w_1**(2*n))/aux_1 ) ** (1/(2*n))
f0 = w_0 / (2*math.pi)

#%%
s = TransferFunction.s

aux = - s**2 / (4 * math.pi**2 * f0**2)
aux = aux**n
#%%
f = np.arange(1e4, 1e7, 10)

aux = f / f0
aux = aux**(2*n)
H = 1 / ( ( 1+aux )**(1/2) )

plt.semilogx(f, H)
plt.axis([1e5, 1e7, H.min()-0.1, H.max()+0.1])
plt.grid()
plt.show()

#%%
H_db = 20*np.log10(H)

plt.semilogx(f, H_db)
plt.axis([1e5, 1e7, H_db.min()-1, H_db.max()+1])
plt.grid()
plt.show()

#%%

x = np.linspace(0, 2*np.pi, 1000)
y1 = np.sin(x)

f = plt.figure()

ax = f.add_subplot(111)

plt.plot(x, y1, '-b', label='sine')

plt.axvline(x=np.pi,color='red')

plt.title('Matplotlib Vertical Line')

plt.xlim(0, 2.0*np.pi)
plt.ylim(-1.5, 1.5)

#plt.savefig('matplotlib_vertical_line.png', bbox_inches='tight')
plt.show()
#%%
fig = plt.figure()

ax = fig.add_subplot(111)

plt.plot(f, H, '-b')

plt.axvline(x=f1, color='red', label='f1')
plt.axvline(x=f2, color='red', label='f2')

#plt.xlim(1e5, 1e7)
#plt.ylim(H.min()-0.1, H.max()+0.1)
plt.axis([1e5, 1e7, H.min()-0.1, H.max()+0.1])
plt.grid()
plt.legend()
plt.show()