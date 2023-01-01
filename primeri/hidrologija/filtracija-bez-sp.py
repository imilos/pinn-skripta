# %%
import numpy as np
import matplotlib.pyplot as plt 
import sciann as sn
from sciann.utils.math import diff, sign, sin, cos, atan, sqrt, abs

# %%
# Osnovni grid
x_data, y_data = np.meshgrid(
    np.linspace(0, 2, 201), 
    np.linspace(0, 2, 201)
)
x_data=x_data.flatten()
y_data=y_data.flatten()
x_data, y_data = np.array(x_data), np.array(y_data)

# %%
# Modeluje se phi(x,y)
x = sn.Variable('x')
y = sn.Variable('y')
phi = sn.Functional('phi', [x,y], 4*[30], 'sigmoid')

# %%
k = 1
TOL = 0.015

# Osnovna jednacina
fun1 = k * (diff(phi, x, order=2) + diff(phi, y, order=2))

# Dirihleovi granicni uslovi
C1 = (1-sign(x - (0+TOL))) * (phi-2)
C2 = (1+sign(x - (2-TOL))) * (phi-1) 

# Njumanovi granicni uslovi
N1 = (1-sign(y - (0+TOL))) * diff(phi,y)
N2 = (1+sign(y - (2-TOL))) * diff(phi,y)

# %%
# FZNN model
m2 = sn.SciModel([x,y], [fun1, C1, C2, N1, N2],  optimizer='Adam')

# Trening
pinn_model = m2.train([x_data, y_data], 5*['zero'], learning_rate=0.001, batch_size=1024, epochs=100, stop_loss_value=1E-15)

# %%
# Test set (x,y)
x_test, y_test = np.meshgrid(
    np.linspace(0, 2, 11), 
    np.linspace(0, 2, 11)
)

# Predikcija
phi_pred = phi.eval(m2, [x_test, y_test])

# %%
x_graph = []
phi_graph = []

for i in range(phi_pred.shape[0]):
    x_graph.append(x_test[5,i])
    phi_graph.append(phi_pred[5,i])

# %%
# Poredjenje SP sa analitickim resenjem
plt.clf()
plt.plot(x_graph, phi_graph, 'o--', label="$\Phi$")
x_axes = np.linspace(0,2,10)
plt.plot(x_axes, 2-1/2*x_axes, "-", linewidth=3, label="Egzaktno rešenje")
plt.xlabel("x [m]")
plt.ylabel("$\Phi$ [m]")
plt.title("2D Filtracija")
plt.legend()
plt.show()


# %%
# Grafik polja potencijala
x_test, y_test = np.meshgrid(
    np.linspace(0, 2, 101), 
    np.linspace(0, 2, 101)
)
phi_pred = phi_pred = phi.eval(m2, [x_test, y_test])
fig, ax = plt.subplots()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title("2D Filtracija - potencijal")
CS = plt.contour(x_test,y_test,phi_pred)
CS = plt.contourf(x_test,y_test,phi_pred)
cbar = fig.colorbar(CS)
cbar.update_ticks()
plt.show()


