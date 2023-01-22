import numpy as np
import matplotlib.pyplot as plt 
import sciann as sn
from numpy import pi
from sciann.utils.math import diff, sign, sin, sqrt
import random

import matplotlib
matplotlib.use('Agg')

# Wave speed
c = 1

# Frequency
omega = 2*pi*4

x = sn.Variable('x')
y = sn.Variable('y')
v, w = sn.Functional (["v", "w"], [x, y], 5*[150] , 'sin')

# Diff. equation
L1 = -omega**2 * v - c**2 * diff(v, x, order=2) - c**2 * diff(v, y, order=2) 
L2 = -omega**2 * w - c**2 * diff(w, x, order=2) - c**2 * diff(w, y, order=2)

TOL = 0.015
#TOL = 0.005

# Dirichlet boundary G1 (y=0 and 0.4<x<0.6)
#C1 = (1 - sign(y - (0 + TOL))) * (1 + sign(x-0.4)) * (1 - sign(x-0.6)) * (1-(v))
#C2 = (1 - sign(y - (0 + TOL))) * (1 + sign(x-0.4)) * (1 - sign(x-0.6)) * (w)
a,b,c,d =  0.39762422, -1.57715550, -0.03696364,  1.60337246
C1 = (1 - sign(y - (a + b*x + c*sqrt(x) + d*x**2 + TOL))) * (1 + sign(x-0.4)) * (1 - sign(x-0.6)) * (1-v) 
C2 = (1 - sign(y - (a + b*x + c*sqrt(x) + d*x**2 + TOL))) * (1 + sign(x-0.4)) * (1 - sign(x-0.6)) * (w-0)

# Upper boundary G2 (where y=1)
C3 =  (1+sign(y - (1-TOL))) * ( c*diff(v,y) - omega*w )
C4 =  (1+sign(y - (1-TOL))) * ( c*diff(w,y) + omega*v )

# Right boundary G2 (where x=1)
C5 =  (1+sign(x - (1-TOL))) * ( c*diff(v,x) - omega*w )
C6 =  (1+sign(x - (1-TOL))) * ( c*diff(w,x) + omega*v )

# Left boundary G2 (where x=0)
C7 =  (1-sign(x - (0+TOL))) * ( -c*diff(v,x) - omega*w )
C8 =  (1-sign(x - (0+TOL))) * ( -c*diff(w,x) + omega*v )

# Bottom boundary G2 (where y=0) and (x<0.4 or x>0.6)
C9 =   (1-sign(y - (0+TOL))) * ( (1 - sign(x-0.4)) + (1 + sign(x-0.6)) ) * ( -c*diff(v,y) - omega*w )
C10 =  (1-sign(y - (0+TOL))) * ( (1 - sign(x-0.4)) + (1 + sign(x-0.6)) ) * ( -c*diff(w,y) + omega*v )

x_data, y_data = [], []


kolokacione_tacke = np.genfromtxt('kolokacione_tacke.txt', delimiter=" ")

for e in kolokacione_tacke:
  ind, x1, y1 = e
  #if y1<0:
  #  continue
  # Prihvati sve tacke blizu granica
  pojas = 1 - np.sign(y1 - (a + b*x1 + c*np.sqrt(x1) + d*x1**2 + TOL))
  if ((x1<0+TOL) or (x1>1-TOL) or (y1<0+TOL) or (y1>1-TOL)) and not ( pojas==0 and (x1>0.4 or x1<0.6) and y1<0+TOL):
  #if (x1<0+TOL) or (x1>1-TOL) or (y1<0+TOL) or (y1>1-TOL):
    x_data.append(x1)
    y_data.append(y1)
  # Ostale tacke prihvati sa nekom verovatnocom
  else:
    if random.random() < 0.99:
      x_data.append(x1)
      y_data.append(y1)


'''
# Training set
x_train, y_train = np.meshgrid(
    np.linspace(0, 1, 200),
    np.linspace(0, 1, 200)
)

for i in range(len(x_train)):
  for j in range(len(y_train)):
    x1,y1 = x_train[i][j],y_train[i][j]
    # Prihvati sve tacke blizu granica
    if (x1<0+TOL) or (x1>1-TOL) or (y1<0+TOL) or (y1>1-TOL):
      x_data.append(x1)
      y_data.append(y1)
    # Ostale tacke prihvati sa nekom verovatnocom
    else:
      if random.random() < 0.2:
        x_data.append(x1)
        y_data.append(y1)
'''


x_data, y_data = np.array(x_data), np.array(y_data)

m = sn.SciModel([x, y], [L1,L2,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10], 'mse', 'Adam')
h = m.train([x_data, y_data], 12*['zero'], learning_rate=0.001, batch_size=1024, epochs=8000, adaptive_weights={'method':'NTK', 'freq':200})

# Test
x_test, y_test = np.meshgrid(
    np.linspace(0, 1, 200), 
    np.linspace(0, 1, 200)
)
v_pred = v.eval(m, [x_test, y_test])
w_pred = w.eval(m, [x_test, y_test])

fig = plt.figure()
plt.pcolor(x_test, y_test, v_pred, cmap='seismic')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
#plt.show()
plt.savefig('socivo-v.png')

fig = plt.figure()
plt.pcolor(x_test, y_test, w_pred, cmap='seismic')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
#plt.show()
plt.savefig('socivo-w.png')

