from flask import Flask, render_template, Response, request, jsonify
import cv2 #OpenCv (procesamiento de imgs)
import mediapipe as mp #rastrear las manos
import numpy as np #manejo de los datos del modelo
from PIL import Image #cambiar tamaño img
from deep_translator import GoogleTranslator
import requests
import json
import time
from io import BytesIO
import tensorflow_hub as hub# Cargar de modelos de AI
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

classes=['a', 'b', 'c', 'd', 'e','eye','f','g','h','i','j','k','l','m','n','o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y','z']

palabra=""
prediccion=""
delay=0
idioma_Traducir='en'
resultado=""

app = Flask(__name__) #Crea una nueva instancia del modilo actual gracias a __name__

cap = cv2.VideoCapture(0)#Inicia la camara web
mp_hands = mp.solutions.hands #uso de las manos
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) #Deteccion y seguimientos de manos por 0.5segundos

#leer la ultima capa o filtro que entrenamos
with custom_object_scope({"KerasLayer":hub.KerasLayer}):
        modelo=load_model('modelo_entrenado.h5') #se integra el modelo entrenado

@app.route('/')#ruta de la app de inicio es el index()
def index():
    return render_template('index.html')

@app.route('/video_feed') #devuelve los fotogramas para verse en la pag web
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#video:
def generate_frames():
    while True:
        global delay
        success, frame = cap.read() #lee el fotograma del video
        if not success: #verifica si se leyo exitoso con un boolean
            break
        
        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB) #poner las img al derecho y no volteadas (se vuelve rgb)
        frame_rgb.flags.writeable = False #no modificar la img mientras se rastrea las manos
        results = hands.process(frame_rgb) #se procesa el rastrador de manos con el frame
        frame_rgb.flags.writeable = True #deja modificar la img
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) #se vuelve BGR
        
        #si se encuentran manos se dibuja los puntos de referencias con MediaPipe:
        if results.multi_hand_landmarks: #verifica si hay atributos
            for hand_landmarks in results.multi_hand_landmarks:
                #se dubuja los puntos en las manos:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                ret, buffer = cv2.imencode('.jpg', frame)
                delay+=1 #usamos delay para que compare las manos despues de x frames
                if delay==12:
                    modelo_senias(buffer)
                    delay=0
               
        ret, buffer = cv2.imencode('.jpg', frame) #el fotograma se codifica en .jpg
        frame_as_bytes = buffer.tobytes() #se convierte el fotograma en bytes

        #se envia el fotograma en tiempo real como video:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_as_bytes + b'\r\n')


def modelo_senias(imagen):
  
    #Categorizar una imagen
    img = Image.open(BytesIO(imagen))
    img = np.array(img).astype(float)/255

    img = cv2.resize(img, (224,224))
    prediccion = modelo.predict(img.reshape(-1, 224, 224, 3))#matriz de predicciones
    #print(prediccion[0])#matriz de predicciones

    #print("////////////////////////////////////////////////////////////")
    #print(np.argmax(prediccion[0], axis=-1))
    indice=np.argmax(prediccion[0], axis=-1)
    #print("la letra es: "+str(classes[indice]))
    #print("////////////////////////////////////////////////////////////")
    
    #Concatenar las letras:
    global palabra
    if len(palabra)==0: #verificar si esta vacio
        palabra += str(classes[indice])
    elif palabra[-1]!=classes[indice]:#verificar que la ultima letra es diferente a la nueva
        palabra += str(classes[indice])
        if len(palabra)>=3 and len(palabra)<=5:
            predecirPalabra(palabra)
    
    #print("--------Esta es la palabra: "+str(palabra))

    if len(palabra)>=5:
        palabra=""


#Mostrar la palabra en la pantalla
@app.route('/')
def mostrar_palabra():
    return render_template('mostrar_palabra.html', palabra=palabra)

@app.route('/actualizar_palabra', methods=['GET'])
def actualizar_palabra():
    global palabra
    mi_variable = palabra
    return jsonify({'palabra': mi_variable})


#Mostrar la prediccion en la pantalla
@app.route('/')
def mostrar_prediccion():
    return render_template('mostrar_prediccion.html', prediccion=prediccion)

@app.route('/actualizar_prediccion', methods=['GET'])
def actualizar_prediccion():
    mi_variable = prediccion
    return jsonify({'prediccion': mi_variable})

def predecirPalabra(pregunta):
    url = "http://127.0.0.1:3002/chat"
    headers = {'Content-Type': 'application/json'}
    data = {'pregunta': "voy a darte unas letras y quiero que prediga lo que intento decir asi no sea exacto, asegurate unicamente darme SOLO las opciones de la palabra, sin escribir nada mas: "+pregunta}

    respuesta = requests.post(url, headers=headers, data=json.dumps(data))
    
    if respuesta.status_code == 200:
        #print(respuesta.json()['respuesta'])
        global prediccion
        prediccion=str(respuesta.json()['respuesta'])
        traducir(prediccion)
    else:
        print(f'Error: {respuesta.status_code}')

@app.route('/')
def mostrar_traduccion():
    return render_template('mostrar_traduccion.html', traduccion=resultado)

@app.route('/actualizar_traductor', methods=['GET'])
def actualizar_traductor():
    return jsonify({'traduccion': resultado})

#traductor:
def traducir(texto):
    global resultado
    traductor = GoogleTranslator(source='es', target=idioma_Traducir)#metodo para traducir
    resultado = traductor.translate(texto)#traducimos el texto y se guarda en resultado
   

#Idioma a traducir
@app.route('/traducir', methods=['POST'])
def obtenerLenguaje():
  idioma = request.form['lenguaje'] #trae el valor de la opcion
  global idioma_Traducir
  #evalua que opcion es:
  if idioma=="1":
      idioma_Traducir='ar' #a la variable global se le asigna el target respectivo al idioma
  elif idioma=="2":
      idioma_Traducir='fr'
  elif idioma=="3":
      idioma_Traducir='pt'
  elif idioma=="4":
      idioma_Traducir='ja'
  elif idioma=="5":
      idioma_Traducir='ru'

  return ('', 204)


if __name__ == '__main__':
    app.run(debug=True) #Ejecuta la app en un navegador web

cap.release()