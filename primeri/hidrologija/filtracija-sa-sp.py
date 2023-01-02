# %%
import numpy as np
import matplotlib.pyplot as plt 
import sciann as sn
from sciann.utils.math import diff, sign, sin, cos, atan, sqrt, abs

# %%
# Osnovni grid
x_data, y_data = np.meshgrid(
    np.linspace(0, 2, 201), 
    np.linspace(0, 2.1, 211)
)
x_data=x_data.flatten()
y_data=y_data.flatten()
x_data, y_data = np.array(x_data), np.array(y_data)

# %%
# Gusci grid
x1, y1 = np.meshgrid(
    np.linspace(0, 2, 801),
    np.linspace(0, 2.1, 841)
)
x1=x1.flatten()
y1=y1.flatten()

# %%
# Odabir tacaka iz gusce mreze
x_region=[]
y_region=[]
for i in range(x1.shape[0]):
    if np.abs(y1[i] - (2-0.5*x1[i])) < 0.15:
        x_region.append(x1[i])
        y_region.append(y1[i])

# Dodavanje tacaka iz gusce mreze na osnovnu
x_data = np.concatenate((x_data, x_region))
y_data = np.concatenate((y_data, y_region))

# %%
# Modeluje se phi(x,y)
x = sn.Variable('x')
y = sn.Variable('y')
phi = sn.Functional('phi', [x,y], 4*[30], 'sigmoid')

# %%
k = 1
TOL = 0.015

# Osnovna jednacina
fun1 =  (y<phi) * k * (diff(phi, x, order=2) + diff(phi, y, order=2))

# Dirihleovi granicni uslovi
C1 = (1-sign(x - (0+TOL))) * (phi-2)
C2 = (1+sign(x - (2-TOL))) * (phi-1) 

# Njumanovi granicni uslovi
N1 = (1-sign(y - (0+TOL))) * diff(phi,y)
N2 = (1+sign(y - (2.1-TOL))) * diff(phi,y)

# %%
# Koeficijent pravca tangente na slobodnu povrsinu
k1 = diff(phi,x)

# Ugao normale na SP
alpha = atan(k1)+np.pi/2
nx = cos(alpha)
ny = sin(alpha)

# Granicni uslov slobodne povrsine
FS1 = (abs(y-phi)<0.009) * k * (diff(phi,x)*nx + diff(phi,y)*ny)

# %%
# FZNN model
m2 = sn.SciModel([x,y], [fun1, C1, C2, N1, N2, FS1],  optimizer='Adam')

# %%

# Trening
pinn_model = m2.train([x_data, y_data], 6*['zero'], learning_rate=0.001, batch_size=1024, epochs=100, stop_loss_value=1E-15)

# %%
# Test set (x,y)
x_test, y_test = np.meshgrid(
    np.linspace(0, 2, 101), 
    np.linspace(0, 2.1, 101)
)
x_test, y_test = np.array(x_test).reshape(-1, 1), np.array(y_test).reshape(-1, 1)

# Predikcija
phi_pred = phi.eval(m2, [x_test, y_test])

# %%
phi_pred.reshape(-1,101)
x_test.reshape(-1,101)
y_test.reshape(-1,101)
x_graph = []
y_graph = []
phi_graph = []

for i in range(phi_pred.shape[0]):
        for j in range(phi_pred.shape[1]):

            # Iznad slobodne povrsine potencijal nema smisla
            if y_test[i,j]>phi_pred[i,j]:
                phi_pred[i,j]=0

            # Gde je y=phi, to je slobodna povrsina
            if np.abs(y_test[i,j]-phi_pred[i,j])<TOL:
                 x_graph.append(x_test[i,j])
                 y_graph.append(y_test[i,j])
                 phi_graph.append(phi_pred[i][j])                 

# %%
# Poredjenje SP sa analitickim resenjem
plt.clf()
plt.plot(x_graph, phi_graph, 'o--', label = "Slobodna površina PINN")
x_axes = np.linspace(0,2,100)
plt.plot(x_axes, np.sqrt(4-1.5*x_axes), "-", linewidth=3, label = "Egzaktno rešenje")
plt.xlabel("x [m]")
plt.ylabel("$\Phi$ [m]")
plt.title("Slobodna površina")
plt.legend()
plt.show()


# %%
# Grafik polja potencijala
x_test, y_test = np.meshgrid(
    np.linspace(0, 2, 101), 
    np.linspace(0, 2.1, 101)
)
phi_pred = phi_pred.reshape(101,-1)
fig, ax = plt.subplots()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title("Potencijal")
CS = plt.contour(x_test,y_test,phi_pred)
CS = plt.contourf(x_test,y_test,phi_pred)
cbar = fig.colorbar(CS)
cbar.update_ticks()
plt.show()


