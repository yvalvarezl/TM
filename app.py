# Título de la app
st.title("🤖 Asistente de Reconocimiento")
st.write("Identificación previa para el uso del dispositivo.")

# --- AQUÍ PONES LA NUEVA IMAGEN ---
try:
    image = Image.open('asistente.jpg')  # Reemplaza 'asistente.jpg' por el nombre de tu archivo cargado
    st.image(image, width=350)
except Exception:
    pass
# ----------------------------------

# Barra lateral informativa
with st.sidebar:
