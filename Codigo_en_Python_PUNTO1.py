# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 16:00:07 2022

@author: RONALD
"""
#Taller 3
#Programacion en lenguajes estadisticos
#Ronald Mateo Ceballos Lozano

#1. Implementar un programa con las siguientes opciones:

#A. Exportar el conjunto de datos gapminder en formato “xlsx”. El 10 %
#de los valores de las columnas "lifeExp", "pop", y "gdpPercap" se debe
#reemplazar de forma aleatoria por valores no asignados NA.

#Instalar e importar la libreria "pandas"
import pandas as pd

#Siguiente a esto, importamos nuestro archivo llamado "archivo_gapminder.xlsx"
#NOTA: REEMPLAZAR LA RUTA USADA ABAJO, POR LA RUTA DEL ARCHIVO UBICADA EN EL NUEVO DISPOSITIVO
datos_originales = pd.read_excel("C:/Users/RONALD/Documents/UNAL 2022/Programacion en Lenguajes Estadisticos/Taller 3/archivo_gapminder.xlsx")
print(datos_originales)

#Verificar el numero de datos de las columnas "lifeExp", "pop" y "gdpPercap"
datos_originales[["lifeExp", "pop", "gdpPercap"]]

#Instalar e importar la libreria "numpy"
import numpy as np

#Indicar por medio de indices los valores NA a reemplazar
indices = np.random.permutation(len(datos_originales))
indices = indices[:170]

#Aplicar los valores NA a cada columna y rectificar que se cumpla en cada una
datos_originales_nan = datos_originales[["lifeExp", "pop", "gdpPercap"]]

datos_originales_nan["lifeExp"][indices] = (pd.NA)
datos_originales_nan["pop"][indices] = (pd.NA)
datos_originales_nan["gdpPercap"][indices] = (pd.NA)

#Verificar si "gapminder" contiene los valores NA
pd.isna(datos_originales_nan["lifeExp"])
pd.isna(datos_originales_nan["pop"])
pd.isna(datos_originales_nan["gdpPercap"])

#B. Importar el archivo gapminder en formato “xlsx”.

#Instalar e importar la libreria "pandas"
#import pandas as pd

#Siguiente a esto, importamos nuestro archivo llamado "archivo_gapminder.xlsx"
#NOTA: Cambiar la ruta del archivo en el dispositivo donde haya sido descargado
datos_originales = pd.read_excel("C:/Users/RONALD/Documents/UNAL 2022/Programacion en Lenguajes Estadisticos/Taller 3/archivo_gapminder.xlsx")
print(datos_originales)

#C. Graficar el diagrama de dispersion lifeEx vs pop

#Instalar y carga la liberia "mathplotlib.pyplot"
import matplotlib.pyplot as plt

plt.scatter(datos_originales["pop"],datos_originales["lifeExp"])
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.title("LifeExp vs Pop")
plt.xlabel('Pop')
plt.ylabel('LifeExp')
plt.show()

#D. Graficar el diagrama de dispersion gdpPercap vs pop

#Instalar y carga la liberia "mathplotlib.pyplot"
import matplotlib.pyplot as plt

plt.scatter(datos_originales["pop"],datos_originales["gdpPercap"])
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.title("GdpPercap vs Pop")
plt.xlabel('Pop')
plt.ylabel('GdpPercap')
plt.show()

#E. Graficar los diagramas de cajas de la variable gdpPercap discriminados por continentes desde 1990 a 2007.

#Instalar y Cargar la libreria "seaborn"
import seaborn as sns

#Filtrar los datos desde los años 1990 a 2007
datos_originales.filter(["year"])
datos_originales_filtrados = datos_originales[(datos_originales.year >= 1990)]

#Verificar si lo datos fueron filtrados
datos_originales_filtrados

#Crear la grafica de cajas de la variable "gdpPercap"
sns.set_style('whitegrid')
ax = sns.boxplot(x='continent',y='gdpPercap',data=datos_originales_filtrados)
ax = sns.stripplot(x="continent", y="gdpPercap",data=datos_originales_filtrados)
