""" Regresion logistica """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap

""" Importación del dataset """
data = pd.read_csv('../datasets/Social_Network_Ads.csv')

""" Seleccion del conjunto de datos en X e y"""
X = data.iloc[:, :-1]
X_cat = X.select_dtypes(include = ['object']).columns.values
X_num = X.select_dtypes(exclude=['object']).columns.values
y = data.iloc[:, -1: ].to_numpy()

""" Eliminacion de valores NaN y conversion de variables categorias a variables dummy """
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(sparse=False), X_cat),
     ('Imputer', SimpleImputer( missing_values=np.nan ,strategy="mean"), X_num)], remainder='passthrough')
X = ct.fit_transform(X)

"""Como solo vamos a ocupar un numero determinado de columnas podemos eliminar algunas,
    un primer enfoque seria eliminarlas despues de leer el dataset es decir en la funcion iloc,
    otra seria hacer todos los calculos para eliminar los valores NaN y realizar las variables dummy
    y eliminar columnas desde la matriz de numpy generadas, aqui usaremos este enfoque """
""" En este caso de la matriz de caracteristicas solo ocuparemos las columnas 2 y 3  """
X = np.delete(X, [0,1,2], 1 )

""" Escalado de variables. Esto se hace porque la columna de EstimateSalary predomina sobre la columna Age.
    En este caso solo se escalaran las columnas del conjunto de variables predictoras"""

"""Division del conjunto de entrenamiento y de test """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state= 0 )

""" Escalado de variables. Esto se hace porque la columna de EstimateSalary predomina sobre la columna Age.
    En este caso solo se escalaran las columnas del conjunto de variables predictoras"""
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""Ajustar el modelo"""
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train.ravel()  )

""" Prediccion del conjunto de valores de test """
y_pred = classifier.predict( X_test )

""" Score """
score = classifier.score(X_test, y_test)
print(f"Score: {score}")

""" Matriz de confusion """
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

# Representación gráfica de los resultados del algoritmo en el Conjunto de Entrenamiento
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set.ravel() == j, 0], X_set[y_set.ravel() == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Clasificador (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()




# Representación gráfica de los resultados del algoritmo en el Conjunto de Testing
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
print()
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set.flatten() == j, 0], X_set[y_set.flatten() == j, 1], color = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Clasificador (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()