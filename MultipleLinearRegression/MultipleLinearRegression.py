""" Regresión lineal multiple """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# importar dataset
data = pd.read_csv('../datasets/50_Startups.csv')

# Seleccion de los nombres de las columnas que contienen datos numericos y categoricos
X = data.iloc[:, :-1]
X_cat = X.select_dtypes(include = ['object']).columns.values
X_num = X.select_dtypes(exclude=['object']).columns.values

# Variable dependiente
y = data.iloc[:, -1: ].to_numpy()

# Transformacion de la matriz hacia una matriz con variables dummy y con eliminacion de valores na
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(sparse=False), X_cat),
     ('Imputer', SimpleImputer( missing_values=np.nan ,strategy="mean"), X_num)], remainder='passthrough')
X = ct.fit_transform(X)

# Eliminacion de una variable dummy
X = X[:, 1:]

# Division del conjunto de datos de entrenamiento y de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 0 )

""" Generacion del modelo de regresion lineal multiple """
regression = LinearRegression()
regression.fit(X_train, y_train)

""" Prediccion de los valores de testing """
y_pred = regression.predict(X_test)

score = regression.score(X_test, y_test)
print(f"Score: { score }")

""" Eliminacion hacia atras  """
""" Primero agregamos una columna de 1's al conjunto de datos original porque el modelo asi lo requiere 
    Agregamos una columna al final y para eso generamos la matriz de 1 que son 50 filas y una columna """
X = np.append(arr = np.ones( (len(X), 1 )).astype(int) , values= X , axis = 1)

""" En primera instancia creamos una matriz con todas las columnas """
X_opt = X[:, [ i for i in range(X.shape[1]) ]]
SL = 0.05

""" En este caso no se necesita dividir el modelo en un conjunto de datos de entrenamiento y de test """
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()

""" Mostramos informacion del modelo 
    La notacion: P> |t | """
print(regression_OLS.summary())

""" El p valor que es mayor a SL es x2 en la primera revision es decir la columna numero 3 ya que la primera es la ordenada al origen.
    Tenemos que volver a ajustar el modelo sin la columna numero 3"""
X_opt = X[:, [0,1,3,4,5]]
""" En este caso no se necesita dividir el modelo en un conjunto de datos de entrenamiento y de test """
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()

""" Mostramos informacion del modelo 
    La notacion: P> |t | """
print(regression_OLS.summary())

""" Ahora es la variable X1 que es la columna numero 2"""
X_opt = X[:, [0,3,4,5]]
""" En este caso no se necesita dividir el modelo en un conjunto de datos de entrenamiento y de test """
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()

""" Mostramos informacion del modelo 
    La notacion: P> |t | """
print(regression_OLS.summary())

""" Ahora es la variable X2 que es la columna numero 2"""
X_opt = X[:, [0,3,5]]
""" En este caso no se necesita dividir el modelo en un conjunto de datos de entrenamiento y de test """
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()

""" Mostramos informacion del modelo 
    La notacion: P> |t | """
print(regression_OLS.summary())

""" Ahora es la variable X2 que es la columna numero 2"""
X_opt = X[:, [0,3]]
""" En este caso no se necesita dividir el modelo en un conjunto de datos de entrenamiento y de test """
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()

""" Mostramos informacion del modelo 
    La notacion: P> |t | """

""" NOTA: LA INSTANCIA AL HACER EL METODO fit() devuelve un nuevo ojeto de RegressionResults por lo cual podemos
    consultar la informacion acerca de los pvalues etc"""

print(regression_OLS.pvalues)
