# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 16:00:59 2022

@author: RONALD
"""

#Taller 3
#Programacion en lenguajes estadisticos
#Ronald Mateo Ceballos Lozano

#2. Implementar un programa con las siguientes opciones:

#A. Cargar dos archivos de datos en formato “csv” llamados “Experimento a.csv”
#y “Experimento b.csv” e indicar si la diferencia en la
#media de los datos es estad´ısticamente significativa.

#Instalar y cargar la libreria "pandas", para importar y leer los formatos .csv.
import pandas as pd

#Una vez hemos obtenido la ruta del archivo la introducimos en una variable nueva
#NOTA: REEMPLAZAR LA RUTA USADA ABAJO, POR LA RUTA DEL ARCHIVO UBICADA EN EL NUEVO DISPOSITIVO

datos_a = pd.read_csv("C:/Users/RONALD/Documents/UNAL 2022/Programacion en Lenguajes Estadisticos/Taller 3/Experimento_a.csv",
                      sep=";",engine="python")

datos_exp_a = pd.DataFrame(datos_a)

datos_b = pd.read_csv("C:/Users/RONALD/Documents/UNAL 2022/Programacion en Lenguajes Estadisticos/Taller 3/Experimento_b.csv",
                          sep=";",engine='python')

datos_exp_b = pd.DataFrame(datos_b)

#Media de Experimento_a
media_a1 = datos_exp_a[["Temperatura"]].mean()
media_a1

media_a2 = datos_exp_a[["Bacterias"]].mean()
media_a2

#Media de Experimento_b
media_b1 = datos_exp_b[["Humedad"]].mean()
media_b1

media_b2 = datos_exp_b[["Temperatura"]].mean() 
media_b2

#B. Cargar dos archivos de datos en formato “csv” llamados “Experimento a.csv” 
#y “Experimento b.csv” y mostrar en pantalla la correlacion
#de Pearson y Spearman de los datos.

#Instalar y cargar la libreria "pandas", para importar y leer los formatos .csv.

import pandas as pd

#Una vez hemos obtenido la ruta del archivo la introducimos en una variable nueva
#NOTA: REEMPLAZAR LA RUTA USADA ABAJO, POR LA RUTA DEL ARCHIVO UBICADA EN EL NUEVO DISPOSITIVO

datos_exp_a = pd.read_csv("C:/Users/RONALD/Documents/UNAL 2022/Programacion en Lenguajes Estadisticos/Taller 3/Experimento_a.csv",
                          sep=";",engine='python')
datos_exp_a

datos_exp_b = pd.read_csv("C:/Users/RONALD/Documents/UNAL 2022/Programacion en Lenguajes Estadisticos/Taller 3/Experimento_b.csv",
                          sep=";",engine='python')
datos_exp_b

#Para calcular los coeficientes de correlacion necesitamos las siguientes librerias:

from scipy.stats import pearsonr
from scipy.stats import spearmanr    

#Ahora usamos el metodo de correlacion de Pearson para el experimento_a

corr_test_pa = pearsonr(x = datos_exp_a['Bacterias'], y =  datos_exp_a['Temperatura'])
print("Coeficiente de correlación de Pearson: ", corr_test_pa[0])

#Ahora usamos el metodo de correlacion de Spearman para el experimento_a

corr_test_sa = spearmanr(datos_exp_a['Bacterias'], datos_exp_a['Temperatura'])
print("Coeficiente de correlación Spearman: ", corr_test_sa[0])

#Ahora usamos el metodo de correlacion de Pearson para el experimento_b

corr_test_pb = pearsonr(x = datos_exp_b['Humedad'], y =  datos_exp_b['Temperatura'])
print("Coeficiente de correlación de Pearson: ", corr_test_pb[0])

#Ahora usamos el metodo de correlacion de Spearman para el experimento_b

corr_test_sb = spearmanr(datos_exp_b['Humedad'], datos_exp_b['Temperatura'])
print("Coeficiente de correlación Spearman: ", corr_test_sb[0])

#C. Cargar dos archivos de datos en formato “csv” llamados“Experimento a.csv”
#y “Experimento b.csv” y graficar el diagrama de dispersion y la linea
#recta que aproxime los datos calculada por una regresion lineal por
#minimos cuadrados.

#Instalar y cargar las librerias "numpy "pandas", para importar y leer los formatos .csv.

import pandas as pd

#Una vez hemos obtenido la ruta del archivo la introducimos en una variable nueva
#NOTA: REEMPLAZAR LA RUTA USADA ABAJO, POR LA RUTA DEL ARCHIVO UBICADA EN EL NUEVO DISPOSITIVO

datos_a = pd.read_csv("C:/Users/RONALD/Documents/UNAL 2022/Programacion en Lenguajes Estadisticos/Taller 3/Experimento_a.csv",
                          sep=";",engine='python')
datos_a

datos_exp_b = pd.read_csv("C:/Users/RONALD/Documents/UNAL 2022/Programacion en Lenguajes Estadisticos/Taller 3/Experimento_b.csv",
                          sep=";",engine='python')
datos_exp_b

# Antes de avanzar tenemos que instalar y cargar las sigueintes librerias para los gráficos

import matplotlib.pyplot as plt
from matplotlib import style


# Al igual con estas otras que nos permitiran construir el modelo lineal

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Procedemos a realizar una configuracion de mathplotlib

plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuramos un alerta

import warnings
warnings.filterwarnings('ignore')

# Empezamos a realizar grafico de dispersión del Experimento_a
fig, ax = plt.subplots(figsize=(6, 3.84))

datos_exp_a.plot(
    x    = 'Temperatura',
    y    = 'Bacterias',
    c    = 'firebrick',
    kind = "scatter",
    ax   = ax
)
ax.set_title('Bacterias vs Temperatura');

# Realizamos la correlacion de Pearson entre 2 variables
corr_test = pearsonr(x = datos_exp_a['Bacterias'], y =  datos_exp_a['Temperatura'])
print("Coeficiente de correlación de Pearson: ", corr_test[0])
print("P-value: ", corr_test[1])

# Dividimos los datos para realizar los test

X = datos_exp_a[['Bacterias']]
y = datos_exp_a['Temperatura']

X_train, X_test, y_train, y_test = train_test_split(
                                        X.values.reshape(-1,1),
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

# Creación del modelo

modelo = LinearRegression()
modelo.fit(X = X_train.reshape(-1, 1), y = y_train)

# Información del modelo
# 
print("Intercept:", modelo.intercept_)
print("Coeficiente:", list(zip(X.columns, modelo.coef_.flatten(), )))
print("Coeficiente de determinación R^2:", modelo.score(X, y))

# Error de test del modelo 

predicciones = modelo.predict(X = X_test)
print(predicciones[0:3,])

rmse = mean_squared_error(
        y_true  = y_test,
        y_pred  = predicciones,
        squared = False
       )
print("")
print(f"El error (rmse) de test es: {rmse}")

# Creación del modelo utilizando matrices como en scikitlearn

# A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo

X_train = sm.add_constant(X_train, prepend=True)
modelo = sm.OLS(endog=y_train, exog=X_train,)
modelo = modelo.fit()
print(modelo.summary())

# Intervalos de confianza para los coeficientes del modelo

modelo.conf_int(alpha=0.05)

# Predicciones con intervalo de confianza del 95%

predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=0.05)
predicciones['x'] = X_train[:, 1]
predicciones['y'] = y_train
predicciones = predicciones.sort_values('x')

# Gráfico del modelo lineal (Regresion Lineal)

fig, ax = plt.subplots(figsize=(6, 3.84))

ax.scatter(predicciones['x'], predicciones['y'], marker='o', color = "gray")
ax.plot(predicciones['x'], predicciones["mean"], linestyle='-', label="OLS")
ax.plot(predicciones['x'], predicciones["mean_ci_lower"], linestyle='--', color='red', label="95% CI")
ax.plot(predicciones['x'], predicciones["mean_ci_upper"], linestyle='--', color='red')
ax.fill_between(predicciones['x'], predicciones["mean_ci_lower"], predicciones["mean_ci_upper"], alpha=0.1)
ax.legend();

# Empezamos a realizar grafico de dispersión del Experimento_b

fig, ax = plt.subplots(figsize=(6, 3.84))

datos_exp_b.plot(
    x    = 'Humedad',
    y    = 'Temperatura',
    c    = 'firebrick',
    kind = "scatter",
    ax   = ax
)
ax.set_title('Humedad vs Temperatura');

# Realizamos la correlacion de Pearson entre 2 variables

corr_test = pearsonr(x = datos_exp_b['Humedad'], y =  datos_exp_b['Temperatura'])
print("Coeficiente de correlación de Pearson: ", corr_test[0])
print("P-value: ", corr_test[1])

# Dividimos los datos para realizar los test

X = datos_exp_b[['Humedad']]
y = datos_exp_b['Temperatura']

X_train, X_test, y_train, y_test = train_test_split(
                                        X.values.reshape(-1,1),
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

# Creación del modelo

modelo = LinearRegression()
modelo.fit(X = X_train.reshape(-1, 1), y = y_train)

# Información del modelo

print("Intercept:", modelo.intercept_)
print("Coeficiente:", list(zip(X.columns, modelo.coef_.flatten(), )))
print("Coeficiente de determinación R^2:", modelo.score(X, y))

# Error de test del modelo 

predicciones = modelo.predict(X = X_test)
print(predicciones[0:3,])

rmse = mean_squared_error(
        y_true  = y_test,
        y_pred  = predicciones,
        squared = False
       )
print("")
print(f"El error (rmse) de test es: {rmse}")

# Creación del modelo utilizando matrices como en scikitlearn

# A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo
X_train = sm.add_constant(X_train, prepend=True)
modelo = sm.OLS(endog=y_train, exog=X_train,)
modelo = modelo.fit()
print(modelo.summary())

# Intervalos de confianza para los coeficientes del modelo

modelo.conf_int(alpha=0.05)

# Predicciones con intervalo de confianza del 95%

predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=0.05)
predicciones['x'] = X_train[:, 1]
predicciones['y'] = y_train
predicciones = predicciones.sort_values('x')

# Gráfico del modelo

fig, ax = plt.subplots(figsize=(6, 3.84))

ax.scatter(predicciones['x'], predicciones['y'], marker='o', color = "gray")
ax.plot(predicciones['x'], predicciones["mean"], linestyle='-', label="OLS")
ax.plot(predicciones['x'], predicciones["mean_ci_lower"], linestyle='--', color='red', label="95% CI")
ax.plot(predicciones['x'], predicciones["mean_ci_upper"], linestyle='--', color='red')
ax.fill_between(predicciones['x'], predicciones["mean_ci_lower"], predicciones["mean_ci_upper"], alpha=0.1)
ax.legend();