import cv2 as cv
import os
import imutils

dataRuta='D:\\Proyecto_Arquitectura_Computadores\\Codigo_fuente\\reconocimiento_facial\\Data'
listaData=os.listdir(dataRuta)  #Ruta de las carpetas
entrenamientoEigenFaceRecognizer=cv.face.EigenFaceRecognizer_create()  #Entrenamiento segun el metodo de Eigen
entrenamientoEigenFaceRecognizer.read('D:\\Proyecto_Arquitectura_Computadores\\Codigo_fuente\\reconocimiento_facial\\EntrenamientoEigenFaceRecognizer.xml')  #Entrenamiento previo
ruidos=cv.CascadeClassifier('C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')  #Entrenamiento ruidos

camara=cv.VideoCapture(0)
#camara=cv.VideoCapture('D:\\Proyecto_Arquitectura_Computadores\\Codigo_fuente\\reconocimiento_facial\\ElonPrueba.mp4')

while True:
    respuesta,captura=camara.read()
    if respuesta==False:break
    grises=grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idcaptura=grises.copy()
    cara=ruidos.detectMultiScale(grises,1.3,5)
    for(x,y,e1,e2) in cara:
        rostrocapturado=idcaptura[y:y+e2,x:x+e1]  #Obteniendo las coordenadas de la captura
        rostrocapturado=cv.resize(rostrocapturado,(160,160),interpolation=cv.INTER_CUBIC)
        resultado=entrenamientoEigenFaceRecognizer.predict(rostrocapturado)  #Comparar con el entrenamiento
        cv.putText(captura,'{}'.format(resultado),(x,y-5),1,1.3,(0,255,0),1,cv.LINE_AA)  #Colocar texto
        if resultado[1]<9000:
            cv.putText(captura, '{}'.format(listaData[resultado[0]]), (x,y-20), 2,1.1,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(captura,(x,y),(x+e1,y+e2),(255,0,0),2)
        else:
            cv.putText(captura,"No encontrado", (x,y-20), 2,0.7,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1,y+e2), (255,0,0),2)

    cv.imshow('Resultados', captura)
    if cv.waitKey(1)==ord('s'):
        break
camara.release()  
cv.destroyAllWindows()