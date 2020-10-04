""" Regresion lineal polinomica """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

""" Importacion del dataset """
data = pd.read_csv('../datasets/Position_Salaries.csv')

""" Seleccion del conjunto de datos en X e y"""
X = data.iloc[:, :-1].select_dtypes(exclude="object").to_numpy()
y = data.iloc[:, -1: ].to_numpy()

""" No es necesario dividir el conjunto de datos en entrenamiento y de test """
""" Ajustando el modelo de regresion lineal """
lin_reg = LinearRegression()
lin_reg.fit(X, y)

""" Ajustando el modelo de regresion lineal polinomica de grado 2 """
poly_reg = PolynomialFeatures(degree=2)
""" X_pol son los terminos de X elevado a la n potencia en donde n es el grado, en este caso es 2 """
X_pol_reg = poly_reg.fit_transform(X)
""" Creamos un modelo de regresion lineal simulando una regresion lineal multiple """
lin_reg_pol = LinearRegression()
lin_reg_pol.fit(X_pol_reg, y )

""" Ajustando el modelo de regresion lineal polinomica de grado 3 """
poly_reg_3 = PolynomialFeatures(degree=3)
""" X_pol son los terminos de X elevado a la n potencia en donde n es el grado, en este caso es 2 """
X_pol_reg_3 = poly_reg_3.fit_transform(X)
""" Creamos un modelo de regresion lineal simulando una regresion lineal multiple """
lin_reg_pol_3 = LinearRegression()
lin_reg_pol_3.fit(X_pol_reg_3, y )

""" Ajustando el modelo de regresion lineal polinomica de grado 4 """
poly_reg_4 = PolynomialFeatures(degree=4)
""" X_pol son los terminos de X elevado a la n potencia en donde n es el grado, en este caso es 2 """
X_pol_reg_4 = poly_reg_4.fit_transform(X)
""" Creamos un modelo de regresion lineal simulando una regresion lineal multiple """
lin_reg_pol_4 = LinearRegression()
lin_reg_pol_4.fit(X_pol_reg_4, y )


"""Subplots"""
""" Visualizacion de los datos del resultado de la regresion lineal"""
fig, axs = plt.subplots(2,2)
axs[0,0].plot(X, y, "r+")
axs[0,0].plot(X, lin_reg.predict(X), color="blue")
axs[0,0].set_xlabel("Posicion del empleado")
axs[0,0].set_ylabel("Sueldo en ($)")
axs[0,0].set_title("Modelo de regresion lineal simple")
# ax1.title("Modelo de regresion lineal simple")
# ax1.xlabel("Posicion del empleado")
# ax1.ylabel("Sueldo en ($)")

""" Visualizacion de los datos del resultado de la regresion polinomica de grado 2"""
axs[0,1].plot(X, y, "r+")
axs[0,1].plot(X, lin_reg_pol.predict(X_pol_reg), color="blue")
axs[0,1].set_xlabel("Posicion del empleado")
axs[0,1].set_ylabel("Sueldo en ($)")
axs[0,1].set_title("Modelo de regresion lineal polinomico")
# ax2.title("Modelo de regresion lineal polinomica")
# ax2.xlabel("Posicion del empleado")
# ax2.ylabel("Sueldo en ($)")

""" Visualizacion de los datos del resultado de la regresion polinomica de grado 3"""
axs[1,0].plot(X, y, "r+")
axs[1,0].plot(X, lin_reg_pol_3.predict(X_pol_reg_3), color="blue")
axs[1,0].set_xlabel("Posicion del empleado")
axs[1,0].set_ylabel("Sueldo en ($)")
axs[1,0].set_title("Modelo de regresion lineal polinomico")

""" Visualizacion de los datos del resultado de la regresion polinomica de grado 3"""
# Definimos mas datos de prediccion
X_aux = np.arange(start= min(X), stop = max(X), step = 0.1 ).reshape(-1, 1)
# Realizamos la transformacion a sus valores poiinomicos de grado 4
X_pol_reg_4 = poly_reg_4.transform(X_aux)
axs[1,1].plot(X, y, "r+")
axs[1,1].plot(X_aux, lin_reg_pol_4.predict(X_pol_reg_4), color="blue")
axs[1,1].set_xlabel("Posicion del empleado")
axs[1,1].set_ylabel("Sueldo en ($)")
axs[1,1].set_title("Modelo de regresion lineal polinomico")
plt.show()

""" Prediccion de un valor """
print(f"Prediccion del valor 6.5 con regresion lineal simple: { float (lin_reg.predict([[6.5]])[0]) }")
print(f"Prediccion del valor 6.5 con regresion lineal polinomica de grado 2: { float ( lin_reg_pol.predict( poly_reg.transform([[6.5]]))[0]) }")
print(f"Prediccion del valor 6.5 con regresion lineal polinomica de grado 3: { float ( lin_reg_pol_3.predict( poly_reg_3.transform([[6.5]]))[0]) }")
print(f"Prediccion del valor 6.5 con regresion lineal polinomica de grado 4: { float(lin_reg_pol_4.predict( poly_reg_4.transform([[6.5]]))[0]) }")