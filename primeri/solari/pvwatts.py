import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from deepxde.backend import tf
import pandas as pd
import pvlib
from pvlib.location import Location
from deepxde.backend import tf
from sklearn.metrics import mean_squared_error

# Ucitaj podatke o vremenu i proizvodnji
solcast_data = pd.read_csv("data_21.08.16_22.10.14.csv", index_col="Time")
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

# Sledecih nedelju dana za testiranje
E_data_test = E_data.loc["2021-11-07":"2021-11-13"]
T_data_test = T_data.loc["2021-11-07":"2021-11-13"]
P_data_test = P_data.loc["2021-11-07":"2021-11-13"]

#
# Parametri modela
#
pdc0 = 0.375 # nominal power [kWh]
Tref = 25.0 # cell reference temperature
gamma_pdc = -0.005 # influence of the cell temperature on PV system
pdc0_inv = 50
eta_inv_nom = 0.96
eta_inv_ref = 0.9637
pac0_inv = eta_inv_nom * pdc0_inv # maximum inverter capacity
a = -2.98 # cell temperature parameter
E0 = 1000 # reference irradiance
deltaT = 1 # cell temperature parameter
num_of_panels = 146 # Broj panela u instalaciji

# Parametar "a" pustamo da se trenira
a_var = dde.Variable(-4.0)

#
# Jednacina koju koristi PINN
#
def pvwatts_eq(x, y):
    Ta = x[:,0:1] # Temperatura vazduha
    E = x[:,1:2] # Ukupna POA osuncanost
    
    Tm = E * tf.exp(a_var) + Ta # Nemamo brzinu vetra
    Tc = Tm + E/E0*deltaT
    P_dc_temp = ((Tc-Tref) * gamma_pdc + 1)
    P_dc = (E * 1.e-03 * pdc0 * P_dc_temp) * num_of_panels
    zeta = (P_dc+1.e-2)/pdc0_inv

    eta = eta_inv_nom/eta_inv_ref * (-0.0162*zeta - 0.0059/zeta + 0.9858)
    eta = tf.maximum(0., tf.sign(eta)) * eta
    ac = tf.minimum(eta*P_dc, pac0_inv)

    return y - ac

#
# Originalni PVWAtts model
#
def orig_pvwatts_model(x):
    Ta = x[:,0:1] # Temperatura
    E = x[:,1:2] # Ukupna POA osuncanost
    
    Tm = E * np.exp(a) + Ta # Nemamo brzinu vetra
    Tc = Tm + E/E0*deltaT
    P_dc_temp = ((Tc-Tref) * gamma_pdc + 1.)
    P_dc = (E * 1.e-03 * pdc0 * P_dc_temp) * num_of_panels
    zeta = (P_dc+1.e-2)/pdc0_inv
    
    eta = eta_inv_nom/eta_inv_ref * (-0.0162*zeta - 0.0059/zeta + 0.9858)
    eta[eta<0] = 0.
    ac = np.minimum(eta*P_dc, pac0_inv)
    
    return ac

#
# Imamo 168 tacaka sa merenjima prozivodnje. Pripremi strukturu za PointSet granicni uslov
#
train_points = np.zeros((168,2))
train_measured_production = np.zeros((168,1))
train_points[:,0] = T_data_train.to_numpy().T
train_points[:,1] = E_data_train.to_numpy().T
train_measured_production[:,0] =  P_data_train.to_numpy().T

#
# Imamo 168 tacaka sa merenjima za narednu nedelju za test
#
test_points = np.zeros((168,2))
test_measured_production = np.zeros((168,1))
test_points[:,0] = T_data_test.to_numpy().T
test_points[:,1] = E_data_test.to_numpy().T
test_measured_production[:,0] =  P_data_test.to_numpy().T

# Minimumi i maksimumi T i E za kreiranje geometrije problema
minT, maxT = min(train_points[:,0]), max(train_points[:,0])
minE, maxE = min(train_points[:,1]), max(train_points[:,1])

geom = dde.geometry.Rectangle([minT, minE], [maxT, maxE])
bc_y = dde.icbc.PointSetBC(train_points, train_measured_production, component=0)

# Isti broj kolokacionih tacaka za jednacinu i za granicne uslove. Moze i drugacije.
data = dde.data.PDE(geom, pvwatts_eq, [bc_y], 168, 168, solution = orig_pvwatts_model, num_test=100)

layer_size = [2] + [30] * 5 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

variable_a = dde.callbacks.VariableValue(a_var, period=1000)
model = dde.Model(data, net)

model.compile(optimizer="adam", lr=0.001, metrics=["l2 relative error"], external_trainable_variables=[a_var])
losshistory, train_state = model.train(iterations=20000, callbacks=[variable_a])

predicted_test = model.predict(test_points)

# Predikcije manje od nule nemaju smisla. Nuluj ih.
predicted_test[predicted_test<0]=0

df = pd.DataFrame(data=test_points, columns=['Ta','E'])
df['P_pred'] = predicted_test
df['P_measured'] = test_measured_production
df['P_pvwatts'] = orig_pvwatts_model(test_points)
df.index = E_data_test.index

df['P_measured'].plot(label='Merena proizvodnja', color='black', linewidth=3)
df['P_pvwatts'].plot(label='Originalni PVWatts model')
df['P_pred'].plot(label="PINN model")
plt.legend(loc="upper left")
plt.ylabel("P [kW]")
plt.title("Modelovanje proizvodnje solarnih panela")
plt.show()

rmse_pinn = mean_squared_error(df['P_measured'], df['P_pred'], squared=False)
rmse_pvwatts = mean_squared_error(df['P_measured'], df['P_pvwatts'], squared=False)
print("PVWATTS RMSE", "%.2f" % rmse_pvwatts)
print("PINN RMSE", "%.2f" % rmse_pinn)
