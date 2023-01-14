import deepxde as dde
import numpy as np

# Frekvencija
n = 2

precision_train = 10
precision_test = 30
weights = 100
iterations = 10000
learning_rate, num_dense_layers, num_dense_nodes, activation = 1e-3, 3, 150, "sin"

# Uvezi sinus
from deepxde.backend import tf
sin = tf.sin

# Osnovna PDE
def pde(x, u):
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)

    f = k0 ** 2 * sin(k0 * x[:, 0:1]) * sin(k0 * x[:, 1:2])
    return -du_xx - du_yy - k0 ** 2 * u - f

# Egzaktno resenje
def func(x):
    return np.sin(k0 * x[:, 0:1]) * np.sin(k0 * x[:, 1:2])

# Da li je kol. tacka na granici?
def boundary(_, on_boundary):
    return on_boundary

# Geometrija jedinicnog kvadrata
geom = dde.geometry.Rectangle([0, 0], [1, 1])
# Talasni broj
k0 = 2 * np.pi * n
# Talasna duzina
wave_len = 1 / n

hx_train = wave_len / precision_train
nx_train = int(1 / hx_train)

hx_test = wave_len / precision_test
nx_test = int(1 / hx_test)

# Dirihleov granicni uslov y=0 na granicama
bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

data = dde.data.PDE(
    geom,
    pde,
    bc,
    num_domain=nx_train ** 2,
    num_boundary=4 * nx_train,
    solution=func,
    num_test=nx_test ** 2,
)

# Mreza i model
net = dde.nn.FNN([2] + [num_dense_nodes] * num_dense_layers + [1], activation, "Glorot uniform")
model = dde.Model(data, net)

# Forsiraj vece tezine za granicne uslove nego za unutrasnjost domena
loss_weights = [1, weights]

model.compile("adam", lr=learning_rate, metrics=["l2 relative error"], loss_weights=loss_weights)

losshistory, train_state = model.train(iterations=iterations)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

