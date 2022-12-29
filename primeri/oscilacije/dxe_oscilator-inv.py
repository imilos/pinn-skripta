import deepxde as dde
from deepxde.backend import tf
import numpy as np
import matplotlib.pyplot as plt

# m * u'' + mu * u' + ku = 0
# m - oscillator mass
# mu -friction coef
# k - spring const

m = 1
mu = 0.6
#k = 2.25
k = dde.Variable(3.)

x0 = 2
v0 = 0

delta = mu / (2*m)
w0 = tf.sqrt(k/m)

# Egzaktno resenje
def func(t):
    return t*1.e-10

# Da li je tacka blizu t=0 (provera pocetnog uslova)
def boundary_l(t, on_initial):
    return on_initial and np.isclose(t[0], 0)

# Jednacina ODE
def Ode(t,x):
        dxdt = dde.grad.jacobian(x,t)
        dxdtt = dde.grad.hessian(x,t)
        return m * dxdtt + mu * dxdt + k*x
    
# x(0)=x0
def bc_func1(inputs, outputs, X):
    return outputs - x0

# x'(0)=v0
def bc_func2(inputs, outputs, X):
    return dde.grad.jacobian(outputs, inputs, i=0,j=None) - v0
    
# Resava se na domenu t=(0,10)
interval = dde.geometry.TimeDomain(0, 10)

# Pocetni uslovi
ic1 = dde.icbc.OperatorBC(interval, bc_func1, boundary_l)
ic2 = dde.icbc.OperatorBC(interval, bc_func2, boundary_l)

bc_x = np.array([1.18, 3.27, 5.37, 7.46, 9.55]).reshape(5,1)
bc_y = np.array([0, 0, 0, 0, 0 ]).reshape(5,1)
ic3 = dde.icbc.PointSetBC(bc_x, bc_y, component=0)

# Definsanje problema, granicnih uslova, broja kolokacionih tacaka
data = dde.data.TimePDE(interval, Ode, [ic1, ic2, ic3], 100, 20, solution=func, num_test=100)
    
layers = [1] + [30] * 2 + [1]
activation = "tanh"
init = "Glorot uniform"
net = dde.nn.FNN(layers, activation, init)

model = dde.Model(data, net)

# Callback funkcija koja stampa varijablu na svakih 1000 epoha
variable = dde.callbacks.VariableValue(k, period=1000)

model.compile("adam", lr=.001, loss_weights=[0.01, 1, 1, 1], metrics=["l2 relative error"], external_trainable_variables=[k])
losshistory, train_state = model.train(epochs=50000, callbacks=[variable])

#dde.saveplot(losshistory, train_state, issave=True, isplot=True)

T = np.linspace(0, 10, 50).reshape(50,1)
x_pred = model.predict(T)

# Egzaktne vrednosti
w0_e = 1.5
delta_e = 0.3
A_e = np.sqrt(26)/5
phi_e = 2*np.arctan(5-np.sqrt(26))
x_e = np.exp(-delta_e*T)*2*A_e*np.cos(phi_e+w0_e*T)

plt.figure()
plt.title("Inverzni problem - nepoznati parametar")
plt.plot(T, x_pred, "o--", label="PINN")
plt.plot(T, x_e, "-", label="Egzaktno")
plt.xlabel("t[s]")
plt.ylabel("x[m]")
plt.legend()
plt.show()
