"""  Maquina de soporte vectorial con regresion lineal """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

""" Importación del dataset """
data = pd.read_csv('../datasets/Position_Salaries.csv')

""" Seleccion del conjunto de datos en X e y"""
X = data.iloc[:, :-1].select_dtypes(exclude="object").to_numpy()
y = data.iloc[:, -1: ].to_numpy()

""" Escalado de las variables """
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_Y.fit_transform(y)

""" Creando una instancia y entrenando el modelo"""
# El metodo ravel es convertir una matriz a un vector
regression = SVR( kernel="rbf" )
regression.fit(X,y.ravel())

""" Visualizacion de los datos escalados """
X_aux = np.arange(start= min(X), stop = max(X), step = 0.1 ).reshape(-1, 1)

_, (ax1, ax2) = plt.subplots(1,2, sharex= 'none', sharey='none')
ax1.plot(X, y, "r+")
ax1.plot(X_aux, regression.predict(X_aux))
ax1.set_title("Modelo de regresion (SVR) con variables escaladas")
ax1.set_xlabel("Posicion del empleado")
ax1.set_ylabel("Sueldo en $")

""" Visualizando los datos sin escalar """
ax2.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(y), "r+")
ax2.plot( sc_X.inverse_transform(X_aux), sc_Y.inverse_transform( regression.predict(X_aux)))
ax2.set_title("Modelo de regresion (SVR) sin variables escaladas")
ax2.set_xlabel("Posicion del empleado")
ax2.set_ylabel("Sueldo en $")
plt.show()
#
""" Prediccion de un valor """
y_pred = regression.predict( sc_X.transform([[6.5]]))
print(f"Valor de predicción para 6.5: { float (sc_Y.inverse_transform(y_pred)) }")
