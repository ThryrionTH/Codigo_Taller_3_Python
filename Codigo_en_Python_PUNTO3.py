# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 16:01:01 2022

@author: RONALD
"""

#Taller 3
#Programacion en lenguajes estadisticos
#Ronald Mateo Ceballos Lozano

#3. Implementar un programa con las siguientes opciones:

#A. Graficar la funcion de densidad de una distribucion uniforme.

#Instalar y cargar las siguientes librerias para realizar las distribuciones:
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# Introducimos los parametros de la distribución Uniforme
uniforme = stats.uniform()
x = np.linspace(uniforme.ppf(0.01),
                uniforme.ppf(0.99), 100)

#Generamos el grafico de la Función de Probabilidad
fp = uniforme.pdf(x)
fig, ax = plt.subplots()
ax.plot(x, fp, '--')
ax.vlines(x, 0, fp, colors='b', lw=5, alpha=0.5)
ax.set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1., 1.2])
plt.title('Distribución Uniforme')
plt.ylabel('Probabilidad')
plt.xlabel('Valores')
plt.show()

# Generamos el Histograma
#Valores Aleatorios
#Se puede modificar en caso de que se requiera
aleatorios = uniforme.rvs(1000)
cuenta, cajas, ignorar = plt.hist(aleatorios, 20)
plt.ylabel('Frequencia')
plt.xlabel('Valores')
plt.title('Histograma Uniforme')
plt.show()

#Por ultimo graficamos la Funcion Densidad

df = pd.DataFrame(x)
df.plot(kind='density')
plt.show()

#B. Graficar la funcion de densidad de una distribucion Bernoulli.

#Instalar y cargar las siguientes librerias para realizar las distribuciones:
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

#Introducimos los parametros de la distribución Bernoulli
p =  0.5
bernoulli = stats.bernoulli(p)
x = np.arange(-1, 3)

#Generamos el grafico de la Función de Probabilidad
fmp = bernoulli.pmf(x)
fig, ax = plt.subplots()
ax.plot(x, fmp, 'bo')
ax.vlines(x, 0, fmp, colors='b', lw=5, alpha=0.5)
ax.set_yticks([0., 0.2, 0.4, 0.6])
plt.title('Distribución Bernoulli')
plt.ylabel('Probabilidad')
plt.xlabel('Valores')
plt.show()

# Generamos el Histograma
#Valores Aleatorios
#Se puede modificar en caso de que se requiera
aleatorios = bernoulli.rvs(1000)  
cuenta, cajas, ignorar = plt.hist(aleatorios, 20)
plt.ylabel('Frequencia')
plt.xlabel('Valores')
plt.title('Histograma Bernoulli')
plt.show()

#Por ultimo graficamos la Funcion Densidad

df = pd.DataFrame(x)
df.plot(kind='density')
plt.show()

#C. Graficar la funcion de densidad de una distribucion Poisson.

#Instalar y cargar las siguientes librerias para realizar las distribuciones:
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

#Introducimos los parametros de la distribución Poisson:
mu =  3.6
poisson = stats.poisson(mu) 
x = np.arange(poisson.ppf(0.01),
              poisson.ppf(0.99))

#Generamos el grafico de la Función de Probabilidad
fmp = poisson.pmf(x) 
plt.plot(x, fmp, '--')
plt.vlines(x, 0, fmp, colors='b', lw=5, alpha=0.5)
plt.title('Distribución Poisson')
plt.ylabel('Probabilidad')
plt.xlabel('Valores')
plt.show()

#Generamos el Histograma
#Valores Aleatorios
#Se puede modificar en caso de que se requiera
aleatorios = poisson.rvs(1000)  
cuenta, cajas, ignorar = plt.hist(aleatorios, 20)
plt.ylabel('Frequencia')
plt.xlabel('Valores')
plt.title('Histograma Poisson')
plt.show()

#Por ultimo graficamos la Funcion Densidad
df = pd.DataFrame(x)
df.plot(kind='density')
plt.show()

#D. Graficar la funcion de densidad de una distribucion Exponencial.

#Instalar y cargar las siguientes librerias para realizar las distribuciones:
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

#Introducimos los parametros de la distribución Exponencial:
exponencial = stats.expon()
x = np.linspace(exponencial.ppf(0.01),
                exponencial.ppf(0.99), 100)

#Generamos el grafico de la Función de Probabilidad
fp = exponencial.pdf(x)
plt.plot(x, fp)
plt.title('Distribución Exponencial')
plt.ylabel('Probabilidad')
plt.xlabel('Valores')
plt.show()

#Generamos el Histograma
#Valores Aleatorios
#Se puede modificar en caso de que se requiera
aleatorios = exponencial.rvs(1000) 
cuenta, cajas, ignorar = plt.hist(aleatorios, 20)
plt.ylabel('Frequencia')
plt.xlabel('Valores')
plt.title('Histograma Exponencial')
plt.show()

#Por ultimo graficamos la Funcion Densidad
df = pd.DataFrame(x)
df.plot(kind='density')
plt.show()
