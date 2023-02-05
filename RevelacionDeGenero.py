#Primero en los archivos de Python se ponen los import. Se puede poner en cualquier parte pero lo recomendable es al inicio.
import numpy as np
import pandas as pd
#Importamos solo una funcion de la libreria, para que nuestro programa ocupe menos espacio
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Leer los datos en un dataframe (dataframe es cuadro de datos)
df = pd.read_csv("Aprendizaje_Supervisado/RevelacionDeGenero.csv")

# Dividir los datos en variables independientes (X) y dependientes (y)
X = df.iloc[:, :-1].values #Esta linea hace referencia a que tome todos los datos menos la ultima columna
y = df.iloc[:, -1].values  #Esta linea hace referencia a que tome todos los datos SOLO de la ultima columna

# Dividir los datos en entrenamiento y prueba.
# X_train, y_train son las variables que se usan para entrenar el modelo
# X_test, y_test son las variables que se usan para probar el modelo
# X son los datos de entrada que les pasas a la funcion (todos los datos independientes, o sea, no la etiqueta que es el objetivo final)
# y son las etiquetas de los datos, o sea, cual es el resultado que les va a dar en la prediccion
# test_size es el porcentaje de datos que se va a utilizar para testear, siendo 0 = 0% y 1.0 = 100%. En este caso 0.2=20% y el resto (80%) es para entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenar el modelo de regresión logística
model = LogisticRegression() #es una técnica de aprendizaje supervisado que se utiliza en el análisis de datos para 
                             #modelar la relación entre una variable dependiente categórica y una o más variables independientes
# Las variables de entrenamiento, las usa para que el modelo entrene
model.fit(X_train, y_train)

# Hacer predicciones sobre los datos de prueba
# Las variables de testeo, las usa para que el modelo haga predicciones
y_pred = model.predict(X_test)

# Evaluar la precisión del modelo
score = accuracy_score(y_test, y_pred) #La precisión se define como el número de predicciones correctas dividido por el número total de predicciones
print("Precisión del modelo:", score)


#Probar si con un nuevo dato, me dice si es masculino o femenino
new_data = [[110, 181]]
prediction = model.predict(new_data)

if prediction[0] == 'mujer':
    print("El genero es femenino")
else:
    print("El genero es masculino")
