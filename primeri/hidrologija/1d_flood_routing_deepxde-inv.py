import deepxde as dde
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deepxde.backend import tf

c = 15 # brzina propagacije talasa
#n = 0.015 # hrapavost kanala
n = dde.Variable(0.02)
Id = 0.005 # nagib dna kanala
B = 15 # poprecni presek
length = 1600
total_time = 1000.0


""" ----------------- HYPERPARAMETERS ----------------- """
layers = [2] + [30] * 4 + [1]
activation = 'tanh'
initializer = 'Glorot uniform'
optimizer = 'rmsprop'
batch_size = 512
num_of_epochs = 30000
learning_rate = 0.001
loss = 'mse'

# Jednacina kontinuiteta
def pde(x, h):
    dh_t = dde.grad.jacobian(h, x, i = 0, j = 1) 
    dh_x = dde.grad.jacobian(h, x, i = 0, j = 0)
    return dh_t + c * dh_x

# Da li je t=0?
def initial_h(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0)

# Da li je x=0?
def boundary_hx0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

# Pocetni uslov za visinu vode x(t=0)
def func_init_h(x):
    return 1.751

# Dirihleov granicni uslov - Profil poplavnog talasa u vremenu
def func_hx0(x):
    t = x[:, 1:2]
    
    Qin = 180 * (1 + (-(np.sign(t - 600) / 2) + 0.5) * np.sin(t *  np.pi / 600))
    a = Qin * n
    b = B * np.sqrt(Id)
    c = a / b
    c = tf.maximum(0.0, tf.sign(c)) * c
    return c**0.6

time_domain = dde.geometry.TimeDomain(0, total_time)
geom_domain = dde.geometry.Interval(0, length)
geotime = dde.geometry.GeometryXTime(geom_domain, time_domain)

# Realizacija granicnog i pocetnog uslova
bc = dde.icbc.DirichletBC(geotime, func = func_hx0, on_boundary = boundary_hx0)
ic = dde.icbc.IC(geotime, func = func_init_h, on_initial = initial_h)

bc_x = np.array([[0,300],[400,320],[800,360],[1200,380],[1600,400]]).reshape(5,2)
bc_y = np.array([2.65,2.65,2.65,2.65,2.65]).reshape(5,1)

ic3 = dde.icbc.PointSetBC(bc_x, bc_y, component=0)

# Callback funkcija koja stampa varijablu na svakih 1000 epoha
variable1 = dde.callbacks.VariableValue(n, period=1000)

data = dde.data.TimePDE(geotime, pde, [bc, ic, ic3], num_domain = 16000, num_boundary = 1000, num_initial = 100, train_distribution = 'uniform')

net = dde.nn.FNN(layers, activation, initializer)

model = dde.Model(data, net)

# Tremniranje RMSProp metodom 
model.compile(optimizer = optimizer, loss = loss, lr = learning_rate, external_trainable_variables=[n])
loss_history, train_state = model.train(epochs = num_of_epochs, display_every = 1000, batch_size = batch_size, callbacks=[variable1])

# Dodatno treniranje L-BFGS-B metodom posle RMSprop optimizacije
model.compile("L-BFGS-B", external_trainable_variables=[n])
loss_history, train_state = model.train(callbacks=[variable1])

delta_t = 20
num_of_points = int(total_time / delta_t) + 1

""" ------------------------ CRTANJE REZULTATA --------------------- """
""" ---------------------------------------------------------------- """
data = pd.DataFrame()
data["time"] = np.linspace(0, total_time, num_of_points)

for i in range(0, length + 1, 400):
    data["xde_" + str(i)] = model.predict(np.vstack((np.full((num_of_points), i), data["time"])).T)[:, 0]

plt.clf()
plt.plot(data["time"], data["xde_0"], label = "h(0, t)")
plt.plot(data["time"], data["xde_400"], label = "h(400m, t)")
plt.plot(data["time"], data["xde_800"], label = "h(800m, t)")
plt.plot(data["time"], data["xde_1200"], label = "h(1200m, t)")
plt.plot(data["time"], data["xde_1600"], label = "h(1600m, t)")
plt.legend()
plt.title("Height of the wave at various points")
plt.xlabel("t [s]")
plt.ylabel("h(x, t) [m]")
plt.show()

