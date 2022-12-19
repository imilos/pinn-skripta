import numpy as np
import matplotlib.pyplot as plt 
import sciann as sn
from numpy import pi
from sciann.utils.math import diff, sign, sin, sqrt, exp
import random

alpha = 1.0

# Inital
t0 = 0.1
s0 = alpha * t0


# Variuable definition
x = sn.Variable('x')
t = sn.Variable('t')
u = sn.Functional (["u"], [x, t], 8*[30] , 'tanh')
s = sn.Functional (["s"], [t], 8*[30] , 'tanh')

# Diff. equation, heat
L1 =  diff(u, t) - alpha * diff(u, x, order=2)

TOLX=0.004
TOLT=0.002

# Stefan condition
C1 = (1/alpha*diff(s, t) + diff(u,x)) * (1 + sign(x - (s-TOLX))) * (1-sign(x-s))

# Pocetno s u trenutku t=t0
C2 = (1-sign(t - (t0+TOLT))) * ( s - s0 )
# Granicni uslov za u kada je x=0
C3 = (1-sign(x - (0 +TOLX))) *  ( u - exp(alpha*t) )

# Temperatura na granici izmedju faza je 1
#C4 = (1+sign(x - (s-TOLX))) * ( u - 1 )
C4 = (1-sign(x - (s+TOLX))) * (1+sign(x-s)) * (u-1)

x_data, t_data = [], []

# Training set
x_train, t_train = np.meshgrid(
    np.linspace(0, 1, 300),
    np.linspace(t0, 0.5, 300)
)

x_data, t_data = np.array(x_train), np.array(t_train)

m = sn.SciModel([x, t], [L1,C1,C2,C3,C4], 'mse', 'Adam')
h = m.train([x_data, t_data], 5*['zero'], learning_rate=0.002, batch_size=1024, epochs=1000, adaptive_weights={'method':'NTK', 'freq':20})

# Test
x_test, t_test = np.meshgrid(
    np.linspace(0, 1, 300), 
    np.linspace(0.01, 0.5, 300)
)
u_pred = u.eval(m, [x_test, t_test])
s_pred = s.eval(m, [x_test, t_test])

s=[]
for e in s_pred:
    s.append(e[0])

fig = plt.figure()
plt.plot(t_test[:,0], s)
plt.savefig('stefan.png')

fig = plt.figure()
plt.plot(x_test[299], u_pred[299])
plt.savefig('u_field.png')

