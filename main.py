from flask import Flask, render_template, Response, request
import cv2 #OpenCv (procesamiento de imgs)
import mediapipe as mp #rastrear las manos
import numpy as np #manejo de los datos del modelo
from PIL import Image #cambiar tamaño img
from deep_translator import GoogleTranslator
import requests
import json
from io import BytesIO
import tensorflow_hub as hub# Cargar de modelos de AI
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope



classes=['a', 'b', 'c', 'd', 'e','eye','f','g','h','i','j','k','l','m','n','o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y','z']
#model = tf.keras.models.load_model('modelo_entrenado.h5') #se integra el modelo entrenado


app = Flask(__name__) #Crea una nueva instancia del modilo actual gracias a __name__

cap = cv2.VideoCapture(0)#Inicia la camara web
mp_hands = mp.solutions.hands #uso de las manos
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) #Deteccion y seguimientos de manos por 0.5segundos

with custom_object_scope({"KerasLayer":hub.KerasLayer}):
        modelo=load_model('modelo_entrenado.h5')

@app.route('/')#ruta de la app de inicio es el index()
def index():
    return render_template('index.html')


def translate_text(text, source_language='es', target_language='en'):
    translator = GoogleTranslator(source=source_language, target=target_language)
    return translator.translate(text)


def generate_frames():
    while True:
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
                modelo_senias(buffer)
               

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
    print(prediccion[0])#matriz de predicciones

    print("////////////////////////////////////////////////////////////")
    print(np.argmax(prediccion[0], axis=-1))
    indice=np.argmax(prediccion[0], axis=-1)
    print("la letra es: "+str(classes[indice]))
    print("////////////////////////////////////////////////////////////")

   # return {
    #    "label":np.argmax(prediccion[0], axis=-1), 
     #   "accuracy":prediccion
    #}

pregunta="bn"

@app.route('/')
def predecirPalabra(pregunta):
    url = "http://127.0.0.1:3002/chat"
    headers = {'Content-Type': 'application/json'}
    data = {'pregunta': "voy a deletrear y quiero que prediga lo que intento decir, asegurate unicamente darme SOLO las opciones de la palabra, sin escribir nada mas: "+pregunta}

    respuesta = requests.post(url, headers=headers, data=json.dumps(data))
    
    if respuesta.status_code == 200:
        print(respuesta.json()['respuesta'])
        print("*********")
        print(str(respuesta.json()['respuesta']))
        
        return str(respuesta.json()['respuesta'])
        #return render_template('index.html',prediccion=respuesta.json()['respuesta'])
    else:
        print(f'Error: {respuesta.status_code}')

@app.route('/video_feed') #devuelve los fotogramas para verse en la pag web
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

prediccion=predecirPalabra(pregunta)
@app.route('/', methods=['POST'])
def traducir():
  
  if request.method == 'POST':
        texto_a_traducir = request.form["texto_a_traducir"]
        traductor = GoogleTranslator(source='es', target='en')
        resultado = traductor.translate(texto_a_traducir)
        return render_template('index.html', resultado=resultado, texto=texto_a_traducir, prediccion=prediccion)


if __name__ == '__main__':
    app.run(debug=True) #Ejecuta la app en un navegador web



cap.release()