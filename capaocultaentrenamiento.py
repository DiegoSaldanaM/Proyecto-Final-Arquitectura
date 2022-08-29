import cv2 as cv
import os 
import numpy as np
from time import time

dataRuta='D:\\Proyecto_Arquitectura_Computadores\\Codigo_fuente\\reconocimiento_facial\\Data'
listaData=os.listdir(dataRuta)  #Ruta de las carpetas
ids=[]
rostrosData=[]
id=0
tiempoInicial=time()  #Tiempo de inicio

for fila in listaData:
    rutacompleta=dataRuta+'\\'+fila  #Ruta de la carpeta individual
    print('Iniciando lectura: ')
    for archivo in os.listdir(rutacompleta):  
        print('Imagenes: ',fila+'\\'+archivo)
        ids.append(id)
        rostrosData.append(cv.imread(rutacompleta+'\\'+archivo,0))  #Datos de los rostros
    
    id=id+1  #Recorre las imagenes
    tiempoFinalLectura=time()
    tiempoTotalLectura=tiempoFinalLectura-tiempoInicial
    print('Tiempo total de lectura: ',tiempoTotalLectura)

entrenamientoEigenFaceRecognizer=cv.face.EigenFaceRecognizer_create()  #Entrenamiento segun el metodo de Eigen
print('Entrenamiento iniciado: ')
entrenamientoEigenFaceRecognizer.train(rostrosData,np.array(ids))   
tiempoFinalEntrenamiento=time()
tiempoTotalEntrenamiento=tiempoFinalEntrenamiento-tiempoTotalLectura
print('Tiempo de entrenamiento total: ',tiempoTotalEntrenamiento)
entrenamientoEigenFaceRecognizer.write('D:\\Proyecto_Arquitectura_Computadores\\Codigo_fuente\\reconocimiento_facial\\EntrenamientoEigenFaceRecognizer.xml')  #Crea el archivo xml
print('Entrenamiento concluido.')