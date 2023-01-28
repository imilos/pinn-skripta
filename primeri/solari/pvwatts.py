import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from deepxde.backend import tf
import pandas as pd
import pvlib
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from deepxde.backend import tf
from sklearn.metrics import mean_squared_error

# Ucitaj podatke o vremenu i proizvodnji
solcast_data = pd.read_csv("./data_21.08.16_22.10.14.csv", index_col="Time")
solcast_data.index = pd.to_datetime(solcast_data.index, dayfirst=True)

# Nagib, azimut i lokacija panela
surface_tilt = 7
surface_azimuth = 290
location = Location(latitude=43.905410, longitude=20.341986, altitude=243, tz="Europe/Belgrade", name="Pons Cacak")

# Pomeri vreme za pola sata, izracunaj poziciju sunca u svakom trenutku
times = solcast_data.index - pd.Timedelta('30min')
solar_position = location.get_solarposition(times)
solar_position.index += pd.Timedelta('30min')

# Novi dataframe sa vrednostima za vrednosti osuncanosti DNI, GHI, DHI
df_poa = pvlib.irradiance.get_total_irradiance(
    surface_tilt=surface_tilt,
    surface_azimuth=surface_azimuth,
    dni=solcast_data['DNI'],
    ghi=solcast_data['GHI'],
    dhi=solcast_data['DHI'],
    solar_zenith=solar_position['apparent_zenith'],
    solar_azimuth=solar_position['azimuth'],
    model='isotropic')

# Izvuci ukupnu POA vrednost iz df_poa
E_data = df_poa["poa_global"]
# Izvuci temperaturu iz podatka o vremenu
T_data = solcast_data["Tamb"]
# Izvuci proizvodnju P
P_data = solcast_data["P"]

# Nedelju dana za treniranje
E_data_train = E_data.loc["2021-10-31":"2021-11-06"]
T_data_train = T_data.loc["2021-10-31":"2021-11-06"]
P_data_train = P_data.loc["2021-10-31":"2021-11-06"]

# Nedelju dana za testiranje
E_data_test = E_data.loc["2021-11-07":"2021-11-13"]
T_data_test = T_data.loc["2021-11-07":"2021-11-13"]
P_data_test = P_data.loc["2021-11-07":"2021-11-13"]

# Parametri
pdc0 = 0.375 # nominal power [Wh]
Tref = 25.0 # cell reference temperature
gamma_pdc = -0.005 # influence of the cell temperature on PV system
pdc0_inv = 50
eta_inv_nom = 0.96
eta_inv_ref = 0.9637
pac0_inv = eta_inv_nom * pdc0_inv # maximum inverter capacity
a = -2.98 # cell temperature parameter
E0 = 1000 # reference irradiance
deltaT = 1 # cell temperature parameter
num_of_panels = 146

# Parametar a pustamo da se trenira
a_var = dde.Variable(-4.0)

def pvwatts_eq(x, y):
    Ta = x[:,0:1] # dry bulb
    E = x[:,1:2] # poa irradiance
    Tm = E * tf.exp(a_var) + Ta # original eq: Tm = E * exp(a+b*WS) + Ta
    #Tm = E * tf.exp(a) + Ta # original eq: Tm = E * exp(a+b*WS) + Ta
    Tc = Tm + E/E0*deltaT
    P_dc_temp = ((Tc-Tref) * gamma_pdc + 1)
    P_dc = (E * 1.e-03 * pdc0 * P_dc_temp) * num_of_panels
    
    zeta = (P_dc+1.e-2)/pdc0_inv
    eta = eta_inv_nom/eta_inv_ref * (-0.0162*zeta - 0.0059/zeta + 0.9858)
    eta = tf.maximum(0.0, tf.sign(eta)) * eta
    ac = tf.minimum(eta*P_dc, pac0_inv)
    return y - 3*ac

def real_pvwatts_model(x):
    Ta = x[:,0:1] # dry bulb
    E = x[:,1:2] # poa irradiance
    
    Tm = E * np.exp(a) + Ta # original eq: Tm = E * exp(a+b*WS) + Ta
    Tc = Tm + E/E0*deltaT
    P_dc_temp = ((Tc-Tref) * gamma_pdc + 1.)
    P_dc = (E * 1.e-03 * pdc0 * P_dc_temp) * num_of_panels
    zeta = (P_dc+1.e-2)/pdc0_inv
    
    eta = eta_inv_nom/eta_inv_ref * (-0.0162*zeta - 0.0059/zeta + 0.9858)
    eta[eta<0] = 0.
    ac = np.minimum(eta*P_dc, pac0_inv)
    
    return 3*ac


# (t, E)
# Kreiraj trening set kolokacionih tacaka
train_points = []
train_measured_production = []
for i in range(E_data_train.shape[0]):
    train_points.append([T_data_train[i], E_data_train[i]])
    train_measured_production.append(float(P_data_train[i]))
train_points = np.array(train_points).reshape(168, 2)
train_measured_production = np.array(train_measured_production).reshape(168, 1)
T_train = train_points[:,0:1]
E_train = train_points[:,1:2]


# (t, E)
# Kreiraj test set kolokacionih tacaka
test_points = []
test_measured_production = []
for i in range(E_data_test.shape[0]):
    test_points.append([T_data_test[i], E_data_test[i]])
    test_measured_production.append(float(P_data_test[i]))
test_points = np.array(test_points).reshape(168, 2)
test_measured_production = np.array(test_measured_production).reshape(168, 1)
T_test = test_points[:,0:1]
E_test = test_points[:,1:2]


minT = min(T_train)[0]
maxT = max(T_train)[0]
minE = min(E_train)[0]
maxE = max(E_train)[0]


geom = dde.geometry.Rectangle([minT, minE], [maxT, maxE])
bc_y = dde.icbc.PointSetBC(train_points, train_measured_production, component=0)
data = dde.data.PDE(geom, pvwatts_eq, [bc_y], 168, 168, solution = real_pvwatts_model, num_test=100)

layer_size = [2] + [30] * 5 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

variable_a = dde.callbacks.VariableValue(a_var, period=1000)
model = dde.Model(data, net)

optimizer = "adam"
learning_rate = 0.001
metric = "l2 relative error"

model.compile(optimizer = optimizer, lr=learning_rate, metrics=[metric], external_trainable_variables=[a_var])
losshistory, train_state = model.train(iterations=20000, callbacks=[variable_a])

predicted_test = model.predict(test_points)

# Predikcije manje od nule nemaju smisla. Nuluj ih
predicted_test[predicted_test<0]=0

df = pd.DataFrame(data=test_points, columns=['Ta','E'])
df['P_pred'] = predicted_test
df['P_measured'] = test_measured_production
df['P_pvwatts'] = real_pvwatts_model(test_points)
df.index = E_data_test.index

df['P_pred'].plot()
df['P_measured'].plot()
plt.legend(loc="upper left")
plt.ylabel("Energy [kWh]")
plt.show()

rmse_pinn = mean_squared_error(test_measured_production, predicted_test, squared=False)
print("PINN RMSE", "%.2f" % rmse_pinn)
