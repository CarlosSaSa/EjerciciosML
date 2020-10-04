""" Modelo de regresion lineal"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# importar dataset
data = pd.read_csv('../datasets/Salary_Data.csv')
X = data.iloc[:, :-1].to_numpy()
y = data.iloc[:, -1: ].to_numpy()

""" En este caso no hay variables categoricas ni filas con valores nan, por lo tanto
    pasamos a realizar la division del conjunto de datos """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state= 0 )

""" En este caso el escalado no es necesario porque las variables no esta siendo predominadas por otras, pasamos a la generacion del modelo el cual
    hace automaticamente el escalado de las variables en este caso con el parametro normalize """
regression = LinearRegression()
regression.fit(X_train, y_train )

# Obteniendo los parametros
print(f"Ordenada al orgien { float(regression.coef_[0])}")
print(f"Pendiente: { float(regression.intercept_) }")

"""Predecir el conjunto de test"""
y_pred = regression.predict( X_test )
# print(f"valores de prediccion: {y_pred}")
# print(f"Valores reales: {y_test}")

"""Visualizacion de los datos"""
plt.plot( X_train, regression.predict(X_train) )
plt.plot(X_train, y_train, 'ro')
plt.plot(X_test, y_test, 'go')
plt.title('Sueldo vs Años de Experiencia (Conjunto de datos de entrenamiento y de test ) ')
plt.xlabel('Años de experiencia')
plt.ylabel("Sueldo (en $")
plt.show()




