import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

# Configuración de la aplicación para celular
st.set_page_config(page_title="Buscador Tienda", page_icon="📸", layout="centered")
st.title("📸 Buscador Visual de Productos")
st.write("Evita subir pisos. Toma una foto y encuentra el precio al instante.")

# Crear la carpeta donde se guardarán las fotos si no existe
DB_FOLDER = "productos_db"
if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

# Cargar el modelo de Inteligencia Artificial (MobileNetV2: ligero y rápido para móviles)
@st.cache_resource
def cargar_modelo():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()
    return model

model = cargar_modelo()

# Transformación para que la IA procese la imagen correctamente
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extraer_caracteristicas(img_pil):
    img_rgb = img_pil.convert("RGB")
    tensor = transform(img_rgb).unsqueeze(0)
    with torch.no_grad():
        features = model.features(tensor)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
    return features.numpy().flatten()

# Pestañas de la aplicación (Buscar y Registrar)
pestana_buscar, pestana_registrar = st.tabs(["🔍 Escanear Producto", "📥 Registrar Producto"])

# --- SECCIÓN PARA REGISTRAR PRODUCTOS ---
with pestana_registrar:
    st.header("Registrar Artículo Nuevo")
    nombre = st.text_input("Nombre del producto (ej: Sarten Rojo Grande)")
    precio = st.text_input("Precio con moneda (ej: 150 Bs)")
    codigo = st.text_input("Código de barras o código interno")
    
    foto_registro = st.camera_input("Toma la foto oficial del producto", key="cam_reg")
    
    if st.button("Guardar en la Base de Datos"):
        if nombre and precio and codigo and foto_registro:
            img = Image.open(foto_registro)
            # Limpiamos el nombre de caracteres que rompan el archivo
            nombre_limpio = nombre.replace("/", "-").replace("\\", "-")
            # El truco: guardamos los datos separados por "___" en el nombre del archivo
            nombre_archivo = f"{precio}___{codigo}___{nombre_limpio}.jpg"
            ruta_guardado = os.path.join(DB_FOLDER, nombre_archivo)
            img.save(ruta_guardado)
            st.success(f"¡'{nombre}' guardado exitosamente!")
        else:
            st.error("Por favor, llena todos los campos y tómale una foto al producto.")

# --- SECCIÓN PARA BUSCAR PRODUCTOS ---
with pestana_buscar:
    st.header("Identificar Producto Sin Código")
    foto_busqueda = st.camera_input("Toma la foto del producto que no tiene etiqueta", key="cam_bus")
    
    if foto_busqueda:
        archivos = [f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if len(archivos) == 0:
            st.warning("La base de datos está vacía. Ve a la pestaña 'Registrar Producto' primero.")
        else:
            with st.spinner("La IA está buscando el producto en la base de datos..."):
                img_busqueda = Image.open(foto_busqueda)
                feat_busqueda = extraer_caracteristicas(img_busqueda)
                
                resultados = []
                for archivo in archivos:
                    ruta_img = os.path.join(DB_FOLDER, archivo)
                    try:
                        img_db = Image.open(ruta_img)
                        feat_db = extraer_caracteristicas(img_db)
                        
                        # Operación matemática para ver qué tan parecidas son las fotos
                        similitud = np.dot(feat_busqueda, feat_db) / (np.linalg.norm(feat_busqueda) * np.linalg.norm(feat_db))
                        resultados.append((similitud, archivo))
                    except:
                        continue
                
                # Ordenar de mayor a menor parecido
                resultados.sort(reverse=True, key=lambda x: x[0])
                
                st.subheader("Resultados encontrados:")
                
                # Mostrar las 3 opciones más parecidas por si acaso
                for i in range(min(3, len(resultados))):
                    sim, archivo = resultados[i]
                    partes = archivo.replace(".jpg", "").split("___")
                    
                    if len(partes) == 3:
                        res_precio, res_codigo, res_nombre = partes
                    else:
                        res_precio, res_codigo, res_nombre = "Error", "Error", archivo
                    
                    # Diseño visual del resultado
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(os.path.join(DB_FOLDER, archivo), use_container_width=True)
                    with col2:
                        st.markdown(f"### **{res_nombre}**")
                        st.markdown(f"💰 **Precio:** {res_precio}")
                        st.markdown(f"🔢 **Código:** {res_codigo}")
                        st.caption(f"Precisión del reconocimiento: {int(sim * 100)}%")
                    st.divider()