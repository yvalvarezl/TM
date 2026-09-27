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

# Título de la app
st.title("🤖 Asistente de Reconocimiento")
st.write("Identificación previa para el uso del dispositivo.")

# Título de la app
st.title("🤖 Asistente de Reconocimiento")
st.write("Identificación previa para el uso del dispositivo.")

# --- AQUÍ PONES LA NUEVA IMAGEN ---
try:
    image = Image.open('watermarked_img_10430299949332107592.jpg')  # Reemplaza 'asistente.jpg' por el nombre de tu archivo cargado
    st.image(image, width=350)
except Exception:
    pass
# ----------------------------------

# Barra lateral informativa
with st.sidebar:

# Barra lateral informativa
with st.sidebar:
    st.header("⚙️ Estado del Sistema")
    st.caption(f"Python v{platform.python_version()}")
    st.write("Clases configuradas:")
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
    prob_yos = float(prediction[0][0])  # Clase 0
    prob_cel = float(prediction[0][1])  # Clase 1

    st.markdown("---")

    # LOGICA DE DETECCIÓN
    if prob_yos > prob_cel and prob_yos > 0.5:
        st.success(f"👋 **¡HOLA YOS!** (Probabilidad: {prob_yos*100:.1f}%)")
        st.info("Te he identificado correctamente. Si quieres habilitar el uso del celular, acércalo a la cámara.")
        
    elif prob_cel > prob_yos and prob_cel > 0.5:
        st.warning(f"📱 **VAS A USAR EL CEL** (Probabilidad: {prob_cel*100:.1f}%)")
        
        # Pregunta interactiva
        st.subheader("¿Yos quiere usar el cel? 🤔")
        opcion = st.radio("Confirma tu acción:", ["Selecciona una opción", "Sí, quiero usarlo", "No, solo estaba probando"])
        
        if opcion == "Sí, quiero usarlo":
            st.balloons()
            st.success("🎉 Acceso concedido al dispositivo. ¡Que lo disfrutes, Yos!")
        elif opcion == "No, solo estaba probando":
            st.info("Entendido, dispositivo en espera.")
            
    else:
        st.error("❓ No logro identificar ni a Yos ni al celular con suficiente certeza. ¡Inténtalo de nuevo!")
