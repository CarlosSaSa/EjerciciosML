""" Arbol de decision """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

""" Importación del dataset """
data = pd.read_csv('../datasets/Position_Salaries.csv')

""" Seleccion del conjunto de datos en X e y"""
X = data.iloc[:, :-1].select_dtypes(exclude="object").to_numpy()
y = data.iloc[:, -1: ].to_numpy()

""" Creacion del modelo """
regression = DecisionTreeRegressor(random_state=0)
regression.fit(X, y)

""" Visualizacion de los datos """
_, (ax1, ax2) = plt.subplots(1,2)
X_aux = np.arange(start= min(X), stop = max(X), step = 0.1 ).reshape(-1, 1)

ax1.plot(X, y, "r+")
ax1.plot(X_aux, regression.predict(X_aux))
ax1.set_title("Modelo de regresion con arboles aleatorios")
ax1.set_xlabel("Posicion del empleado")
ax1.set_ylabel("Sueldo en $")

ax2.plot( X, y , "r+")
ax2.plot( X, regression.predict(X))
ax2.set_title("Modelo de regresion con arboles aleatorios")
ax2.set_xlabel("Posicion del empleado")
ax2.set_ylabel("Sueldo en $")
plt.show()

""" Prediccion de un valor """
y_pred = regression.predict( [[6.5]])
print(f"Valor de predicción para 6.5: { float (y_pred) }")