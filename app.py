import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
import platform

# Configuración de página
st.set_page_config(page_title="Registro con Validación Facial", page_icon="🔐", layout="centered")

# Inicialización de estado de autenticación
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Cargar modelo y etiquetas
@st.cache_resource
def get_model():
    model = load_model('keras_model.h5')
    with open('labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

model, labels = get_model()

# Título y presentación
st.title("🔐 Registro de Usuario")
st.write("Para iniciar el registro, primero debemos verificar tu identidad usando la cámara.")
st.caption(f"Versión de Python: {platform.python_version()}")

# Imagen del encabezado
try:
    image = Image.open('OIG5.jpg')
    st.image(image, width=350)
except Exception:
    pass

# Barra lateral informativa
with st.sidebar:
    st.header("⚙️ Verificación Biométrica")
    st.subheader("Este sistema valida si eres el usuario autorizado antes de habilitar el formulario.")

st.markdown("---")

# ---------------------------------------------------------
# PASO 1: VALIDACIÓN DE IDENTIDAD CON TEACHABLE MACHINE
# ---------------------------------------------------------
st.header("Paso 1: Validación de Identidad")

img_file_buffer = st.camera_input("Toma una foto para validar tu acceso")

if img_file_buffer is not None:
    # Procesar la imagen tomada por la cámara
    img = Image.open(img_file_buffer)
    size = (224, 224)
    img_resized = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img_resized)
    
    # Normalizar imagen para Teachable Machine
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predicción del modelo
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = labels[index]
    confidence_score = float(prediction[0][index])

    # Extraer el nombre de la clase
    user_detected = class_name.split(' ', 1)[-1] if ' ' in class_name else class_name

    # Condición de validación (Ajusta 'Yoselin' o el nombre exacto de tu clase autorizada)
    if confidence_score > 0.70 and ("Yoselin" in user_detected or "1" in class_name):
        st.session_state.authenticated = True
        st.success(f"✅ ¡Identidad verificada con éxito! Bienvenido/a, **{user_detected}** ({confidence_score*100:.1f}% de confianza).")
    else:
        st.session_state.authenticated = False
        st.error(f"❌ Acceso denegado. Se detectó **{user_detected}** ({confidence_score*100:.1f}% de confianza). Se requiere usuario autorizado.")

st.markdown("---")

# ---------------------------------------------------------
# PASO 2: FORMULARIO DE REGISTRO
# ---------------------------------------------------------
st.header("Paso 2: Datos de Registro")

if st.session_state.authenticated:
    st.success("🔓 Acceso Autorizado — Completa los siguientes datos de registro:")
    
    with st.form("registro_usuario"):
        col1, col2 = st.columns(2)
        with col1:
            nombre = st.text_input("Nombre Completo")
            correo = st.text_input("Correo Electrónico")
        with col2:
            documento = st.text_input("Número de Documento")
            rol = st.selectbox("Rol", ["Estudiante", "Docente", "Invitado"])
            
        biografia = st.text_area("Perfil / Observaciones")
        
        enviado = st.form_submit_button("Completar Registro")
        if enviado:
            st.balloons()
            st.success("🎉 ¡Registro completado y guardado correctamente!")
else:
    st.warning("⚠️ Debes validarte en la cámara como usuario autorizado para activar el formulario.")
