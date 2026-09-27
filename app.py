import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
import platform

# Configuración de página
st.set_page_config(page_title="Reconocimiento Biométrico · Yos", page_icon="📱", layout="centered")

# Cargar modelo y etiquetas
@st.cache_resource
def get_model():
    model = load_model('keras_model.h5')
    with open('labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

model, labels = get_model()

# Título y presentación
st.title("🤖 Asistente de Reconocimiento")
st.write("Identificación previa para el uso del dispositivo.")
st.caption(f"Versión de Python: {platform.python_version()}")

# Imagen de encabezado
try:
    image = Image.open('OIG5.jpg')
    st.image(image, width=320)
except Exception:
    pass

# Barra lateral informativa
with st.sidebar:
    st.header("⚙️ Estado del Sistema")
    st.write("Clases configuradas en Teachable Machine:")
    for l in labels:
        st.write(f"- `{l}`")

st.markdown("---")

# Cámara de detección
st.subheader("📸 Verificación en Cámara")
img_file_buffer = st.camera_input("Pónte frente a la cámara (o acerca el celular)")

if img_file_buffer is not None:
    # Procesar imagen
    img = Image.open(img_file_buffer)
    size = (224, 224)
    img_resized = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img_resized)
    
    # Normalización para Teachable Machine
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predicción
    prediction = model.predict(data)
    prob_yos = float(prediction[0][0])  # Clase 0 (Yos)
    prob_cel = float(prediction[0][1])  # Clase 1 (Cel)

    st.markdown("---")
    
    # Mostrar probabilidades en vivo
    st.write(f"📊 **Detección:** Yos: {prob_yos*100:.1f}% | Celular: {prob_cel*100:.1f}%")

    # LÓGICA DE DETECCIÓN
    if prob_yos > prob_cel and prob_yos > 0.5:
        st.success(f"👋 **¡HOLA YOS!**")
        st.info("Te he identificado correctamente. Si quieres habilitar el uso del celular, acércalo a la cámara.")
        
    elif prob_cel > prob_yos and prob_cel > 0.5:
        st.warning(f"📱 **VAS A USAR EL CEL**")
        
        # Pregunta interactiva
        st.subheader("¿Yos quiere usar el cel? 🤔")
        opcion = st.radio("Confirma tu acción:", ["Selecciona una opción", "Sí, quiero usarlo", "No, solo estaba probando"])
        
        if opcion == "Sí, quiero usarlo":
            st.balloons()
            st.success("🎉 Acceso concedido al dispositivo. ¡Que lo disfrutes, Yos!")
        elif opcion == "No, solo estaba probando":
            st.info("Entendido, dispositivo en espera.")
            
    else:
        st.error("❓ No logro identificar claramente ni a Yos ni al celular. ¡Inténtalo de nuevo!")
