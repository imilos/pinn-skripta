import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

''' fixed parameters ''' 
f1_0 = 43.3 
h = 15.6
g1 = 10.0
g2 = 209.0
fzah = 4.0
a = 1.

# Verovatnoca kacenja
def f(x,a):
    return (1+tf.sign(x)) * (1-tf.sign(x-h)) * (f1_0*a*x/h) * 0.25

# Verovatnoca raskacivanja
def g(x):
    return 0.5 * (1-tf.sign(x)) * g2 + \
           0.25 * (1+tf.sign(x)) * (1-tf.sign(x-h)) * (g1*x/h) + \
           0.5 * (1+tf.sign(x-h)) * (fzah*g1*x/h)

# n = n(x,t)
def pde(x, n):
    dn_dt = dde.grad.jacobian(n, x, i=0, j=1)
    loss = dn_dt - (1.0-n) * f(x[:,0:1],a) + n*g(x[:,0:1])
    # Obezbedi pozitivna resenja
    return loss + n*(1-tf.sign(n))

# Computational geometry
geom = dde.geometry.Interval(-20.8, 63)
timedomain = dde.geometry.TimeDomain(0, 0.4)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Pocetni uslovi
ic1 = dde.icbc.IC(geomtime, lambda x: 0.0, lambda _, on_initial: on_initial)

data = dde.data.TimePDE(geomtime, pde, [ic1], num_domain=10000, num_boundary=100, num_initial=500)
net = dde.nn.FNN([2] + [40] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
model.train(100000)
model.compile("L-BFGS", loss_weights=[1.e-1, 1])
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=False, isplot=True)

# 2D grafik
x_test, t_test = np.meshgrid(
                np.linspace(-20.8, 63, 200),
                np.linspace(0, 0.4, 100))

X = np.vstack((np.ravel(x_test), np.ravel(t_test))).T
n_pred = model.predict(X)

for i in [10, 20, 50, 99]:
    t = str(i*0.4/200)
    plt.plot(X[200*i:200*i+200,0], n_pred[200*i:i*200+200], label='t='+t)
    
plt.title("Huxley - izometrijski slucaj")
plt.xlabel("x [nm]")
plt.ylabel("n(x)")
plt.legend()
plt.show()

