import numpy as np
import matplotlib.pyplot as plt 
import sciann as sn
from sciann.utils.math import diff, sign, sin, sqrt, exp
from numpy import pi

x = sn.Variable('x')
t = sn.Variable('t')
u = sn.Functional('u', [x,t], 3*[20], 'tanh')
#alpha = sn.Parameter(0.2, inputs=[x,t], name="alpha")
alpha = 0.4

L1 = diff(u, t) - alpha * diff(u, x, order=2)

TOL = 0.011
TOLT=0.0011
C1 = (1-sign(t - TOLT)) * (u - sin(pi*x))
C2 = (1-sign(x - (0+TOL))) * (u)
C3 = (1+sign(x - (1-TOL))) * (u)
#C4 = (1 + sign(t-0.201)) * (1 - sign(t-0.219)) * (1 + sign(x-0.491)) * (1 - sign(x-0.509)) * (u-0.8)

m = sn.SciModel([x, t], [L1, C1, C2, C3], 'mse', 'Adam')

x_data, t_data = np.meshgrid(
    np.linspace(0, 1, 101), 
    np.linspace(0, 0.1, 101)
)

h = m.train([x_data, t_data], 4*['zero'], learning_rate=0.002, batch_size=128, epochs=500, 
    adaptive_weights={'method':'NTK', 'freq':100})

# Test
nx, nt = 20, 10
x_test, t_test = np.meshgrid(
    np.linspace(0.01, 0.99, nx+1), 
    np.linspace(0.01, 0.1, nt+1)
)
u_pred = u.eval(m, [x_test, t_test])

#print(alpha.value)

plt.plot(x_test[nt], u_pred[nt], 'o--', label='PINN')
ue = np.sin(np.pi * x_test[nt]) * np.exp(-0.1 * alpha * (np.pi)**2)
rmse = np.sqrt(sum((u_pred[nt]-ue)**2 / len(u_pred[nt])))
print("RMSE:", rmse)
plt.plot(x_test[nt], ue, '-', label='Exact')
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.show()

