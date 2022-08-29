import cv2 as cv
import os
import imutils

modelo='FotosDiego'  #Carpeta de capturas propias
ruta1='D:\\Proyecto_Arquitectura_Computadores\\Codigo_fuente\\reconocimiento_facial'  #Ruta del proyecto
rutacompleta=ruta1+'\\'+modelo  #Ruta de la carpeta de capturas

if not os.path.exists(rutacompleta):  #Comprueba la existencia de la carpeta
    os.makedirs(rutacompleta)

camara=cv.VideoCapture(0)  #Abre la camara en vivo o la ruta de la imagen
ruidos=cv.CascadeClassifier('C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
id=0  #Valor para la captura de imagenes

while True:
    respuesta,captura=camara.read()  #Camara capturando
    if respuesta==False:break  #Si la camara no funciona

    captura=imutils.resize(captura,width=640)

    grises=cv.cvtColor(captura,cv.COLOR_BGR2GRAY)  #Pasa a escala de grises
    idcaptura=captura.copy()  #Valor para los puntos de la captura

    cara=ruidos.detectMultiScale(grises,1.3,5)  #Detecta rostros

    for(x,y,e1,e2) in cara:  #Recorre todo el video
        cv.rectangle(captura,(x,y),(x+e1,y+e2),(0,255,0),2)  #Marca un rectangulo para el rostro
        rostrocapturado=idcaptura[y:y+e2,x:x+e1]  #Obteniendo las coordenadas de la captura
        rostrocapturado=cv.resize(rostrocapturado,(160,160),interpolation=cv.INTER_CUBIC)
        cv.imwrite(rutacompleta+'\\imagen_{}.jpg'.format(id),rostrocapturado)  #Guardar la imagen
        id=id+1

    cv.imshow("Resultado rostro",captura)

    if id==351:  #Deteniene la toma de imagenes
        break

camara.release()
cv.destroyAllWindows()  #Cerrar ventanas