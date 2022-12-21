import numpy as np
import matplotlib.pyplot as plt 
import sciann as sn
from numpy import pi
from sciann.utils.math import diff, sign, sin, sqrt, exp
import random

alpha = 1.0

# Pocetni uslovi
t0 = 0.1
s0 = alpha * t0

# Varijable
x = sn.Variable('x')
t = sn.Variable('t')
u = sn.Functional (["u"], [x, t], 3*[30] , 'tanh')
s = sn.Functional (["s"], [t], 3*[30] , 'tanh')

# Glavna dif. jednacina
L1 =  diff(u, t) - alpha * diff(u, x, order=2)

TOLX=0.004
TOLT=0.002

# Stefanov uslov
C1 = (1/alpha*diff(s, t) + diff(u,x)) * (1 + sign(x - (s-TOLX))) * (1-sign(x-s))
# Pocetno s u trenutku t=t0
C2 = ( s - s0 ) * (1-sign(t - (t0+TOLT)))
# Granicni uslov za u kada je x=0
C3 = ( u - exp(alpha*t) ) * (1-sign(x - (0 +TOLX)))
# Temperatura na granici izmedju faza je 1
C4 = (u-1) * (1-sign(x - (s+TOLX))) * (1+sign(x-s))

x_data, t_data = [], []

# Trening skup
x_train, t_train = np.meshgrid(
    np.linspace(0, 1, 300),
    np.linspace(t0, 0.5, 300)
)

x_data, t_data = np.array(x_train), np.array(t_train)

m = sn.SciModel([x, t], [L1,C1,C2,C3,C4], 'mse', 'Adam')
h = m.train([x_data, t_data], 5*['zero'], learning_rate=0.002, batch_size=1024, epochs=20, adaptive_weights={'method':'NTK', 'freq':20})

# Test
x_test, t_test = np.meshgrid(
    np.linspace(0, 1, 30), 
    np.linspace(0.01, 0.5, 30)
)
u_pred = u.eval(m, [x_test, t_test])
s_pred = s.eval(m, [x_test, t_test])

s=[]
for e in s_pred:
    s.append(e[0])

fig = plt.figure()
t1 = t_test[:,0]
plt.plot(t1, s, 'o--', label='s (PINN)')
plt.plot(t1, alpha * t1, '-', label='s (Egzaktno)')
plt.xlabel("t")
plt.ylabel("s")
plt.title("Položaj granice između faza")
plt.show()

fig = plt.figure()
plt.plot(x_test[29], u_pred[29], 'o--', label='u (PINN)')
x1 = x_test[29]
plt.plot(x1, np.exp(alpha*0.5)-x1, '-', label='u (Egzaktno)')
plt.xlabel("x")
plt.ylabel("u")
plt.title("Polje temperature")
plt.show()
