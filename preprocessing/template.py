import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Eliminar notacion exponencial
# np.set_printoptions(suppress=True)

# importar dataset
data = pd.read_csv('../datasets/Data.csv')

""" 
    Seleccionando los datos para la variable X, con iloc seleccionamos todas las filas y todas las columnas menos la ultima
    esto nos regresa un dataframe con el cual con el metodo to_numpy() convertimos el dataframe a una matriz de numpy
    en el cual las filas son las filas del dataframe
 """

# X = data.iloc[:, :-1].to_numpy()
# y = data.iloc[:, -1:].to_numpy()
"""
     1. Seleccionando las columnas con valores categoricos, el atributo columns devuelve un Index por cual con el metodo values obtenemos una lista con los nombres de las columnas
"""
X = data.iloc[:, :-1]
X_cat = data.select_dtypes(include = ['object']).iloc[:, :-1].columns.values
X_num = data.select_dtypes(exclude=['object']).columns.values
y = data.iloc[:, -1: ].to_numpy()


# Tratamiento de los datos NaN, se empleará la estrategia la cual reemplazará los NaN por la media de la columna
"""
    1- Con OneHotEncoder indicamos que a partir del dataframe se van a seleccionar las columnas que estan indicadas con X_cat, de estas columnas se van a codificar en variables dummy, esto devolvera
       una matriz con las filas codificadas y con columnas N donde N es el numero de diferentes categorias
    2- Con Imputer se van a remplazar los variables NaN de las columnas dadas por X_num, esto devolvera una matriz con las mismas filas y las mismas columnas pero con los valores nan reemplazados por la media
    3- Al final la matriz de variables dummy y la matriz de Imputer se van a concatenar en una sola
"""
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(sparse=False), X_cat),
     ('Imputer', SimpleImputer( missing_values=np.nan ,strategy="mean"), X_num)], remainder='passthrough')
X = ct.fit_transform(X)

""" Codificacion de la variable Y """
y_dummy = OneHotEncoder(sparse=False)
# Matriz de variables dummy
y = y_dummy.fit_transform(y)

# Division del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 1 )
# print(f"X_train: { X_train }")
# print(f"X_test: { X_test }")
# print(f"y_train: { y_train }")
# print(f"y_test: { y_test }")

""" Escalado de datos.
    La variable Y en algunos caso si se escala """
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)








