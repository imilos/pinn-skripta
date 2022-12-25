import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# m * u'' + mu * u' + ku = 0
# m - oscillator mass
# mu -friction coef
# k - spring const

m = 1
mu = 0.1
k = 2

x0 = 0
v0 = 2

delta = mu / (2*m)
w0 = np.sqrt(k/m)

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

# Definsanje problema, granicnih uslova, broja kolokacionih tacaka
data = dde.data.TimePDE(interval, Ode, [ic1, ic2], 100, 20, solution=func, num_test=100)
    
layers = [1] + [30] * 2 + [1]
activation = "tanh"
init = "Glorot uniform"
net = dde.nn.FNN(layers, activation, init)

model = dde.Model(data, net)

model.compile("adam", lr=.001, loss_weights=[0.01, 1, 1], metrics=["l2 relative error"])
losshistory, train_state = model.train(epochs=10000)

#dde.saveplot(losshistory, train_state, issave=True, isplot=True)

T = np.linspace(0, 10, 100).reshape(100,1)
x_pred = model.predict(T)
np.savetxt("spring-10000.txt", x_pred)

plt.figure()
plt.plot(T, x_pred, "o--")
plt.show()

