import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
import requests

#CARGAR DATA
enlace=r"https://upcedupe-my.sharepoint.com/:x:/g/personal/u20211c225_upc_edu_pe/ERXL-HiGV2JPvAwYixH8P0sBysPvv9lj96VIVKXU4Mxcxw?download=1" #download=1
#Simula navegador
headers={"User-Agent": "Mozilla/5.0"}
#Hace la solicitud al servidor
response=requests.get(enlace, headers=headers)
archivo=BytesIO(response.content)
read=pd.ExcelFile(archivo)
n_muestras=20
input=[]
output=[]
for i in range(n_muestras):
    name_hoja=f"M{i+1}"
    hoja=read.parse(name_hoja)
    granulometria=np.array(hoja["Granulometria"][0:20])
    Q_in=hoja["Q"][0]
    n_manning=hoja["n_manning"][0]
    dx=hoja["dx"][0]
    Temp=hoja["T"][0]
    cordenadax=np.array(hoja["Ubicación"][0:101])   
    #Bucle
    for delta_t in range(1,24):     #1h a 24h
        for k in range(24-delta_t):     #para no pasar a D25
            Z=np.array(hoja[f"D{k+1}"][0:101][::-1])
            S=-np.gradient(Z,dx)
            Contenidos_data_E=np.concatenate([Z.T,cordenadax.T,S.T,granulometria.T,[Q_in],[n_manning],[Temp],[delta_t]],axis=0)
            Salida=np.array(hoja[f"D{k+1+delta_t}"][0:101][::-1]).T
            output.append(Salida)
            input.append(Contenidos_data_E)
input=np.vstack(input)
output=np.vstack(output)

#Escalador
#Creacion de escaladores Input
scaler_D_MMS_I=MinMaxScaler()
scaler_D_SS_I=StandardScaler()
scaler_D_RS_I=RobustScaler()
scaler_D_NN_L1_I=Normalizer(norm="l1")
scaler_D_NN_L2_I=Normalizer(norm="l2")
scaler_D_NN_max_I=Normalizer(norm="max")

#Creacion de escaladores Output
scaler_D_MMS_O=MinMaxScaler()
scaler_D_SS_O=StandardScaler()
scaler_D_RS_O=RobustScaler()
scaler_D_NN_L1_O=Normalizer(norm="l1")
scaler_D_NN_L2_O=Normalizer(norm="l2")
scaler_D_NN_max_O=Normalizer(norm="max")

#Normalizar
IRS=scaler_D_RS_I.fit_transform(input)
ORS=scaler_D_RS_O.fit_transform(output)

ISS=scaler_D_SS_I.fit_transform(input)
OSS=scaler_D_SS_O.fit_transform(output)

IMM=scaler_D_MMS_I.fit_transform(input)
OMM=scaler_D_MMS_O.fit_transform(output)

INN1=scaler_D_NN_L1_I.fit_transform(input)
ONN1=scaler_D_NN_L1_O.fit_transform(output)

INN2=scaler_D_NN_L2_I.fit_transform(input)
ONN2=scaler_D_NN_L2_O.fit_transform(output)

INNmax=scaler_D_NN_max_I.fit_transform(input)
ONNmax=scaler_D_NN_max_O.fit_transform(output)



#python -m streamlit run app.py

st.set_page_config(
    page_title="RNA_MLP_EROSION",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 RNA MLP PARA LA PREDICCIÓN DE EROSIÓN EN UN CANAL RECTANGULAR")

with st.expander("🔬 Descripción del proyecto",expanded=True):
    st.markdown("""
    <div style='text-align: justify; font-size: 16px;'>
    Este proyecto consiste en un estudio experimental donde se entrena una <b>red neuronal artificial tipo MLP (Perceptrón Multicapa)</b> para predecir la <b>socavación en un canal rectangular</b> de laboratorio con condiciones controladas.

    El canal tiene dimensiones fijas: <b>ancho (b) = 0.5 m</b>, <b>altura (h) = 0.7 m</b> y <b>longitud (L) = 10 m</b>. A lo largo de este canal se simulan distintos escenarios de flujo, variando parámetros como el <b>caudal</b>, la <b>pendiente</b> y otras condiciones hidráulicas, con el fin de observar su influencia en el desarrollo de la erosión en el lecho.

    Los datos obtenidos se utilizan para entrenar el modelo MLP, que busca reconocer patrones no lineales entre las condiciones iniciales del flujo y la magnitud de la socavación resultante. Este enfoque permite evaluar el potencial de las redes neuronales como herramienta de análisis en problemas hidrodinámicos complejos, dentro de un entorno de laboratorio bien definido.
    </div>
    """, unsafe_allow_html=True)

with st.expander("🔬 Metricas de Entrenamiento",expanded=True):
    st.markdown("""
    <div style='text-align: center; font-size: 18px;'>
    📈 R² = 99.96693%<br>
    📉 MAPE = 0.30369%<br>
    🕯️ Loss: 0.002697428222745657
    </div>
    """, unsafe_allow_html=True)


with st.expander("🔎 Parametros",expanded=True):
    #Crear un DataFrame de ejemplo
    data=pd.DataFrame({
        "Caudal (m³/s)": [0.05],
        "T (C°)": [20.0],
        "Manning":[0.025],
        "X-inicio (m)":[0.0],
        "X-final (m)":[10.0001],
        "Zb-inicial (m)":[40.3],
        "Zb-final (m)":[40.0001],
        "Tiempo (Simulado) (Hr.)":[1.0]
    })

    #Mostrar la tabla editable /num_rows="dynamic",
    edited_data=st.data_editor(
        data,
        use_container_width=True,
        column_config={
            "Caudal (m³/s)": st.column_config.NumberColumn("Caudal (m³/s)", min_value=0.,step=0.00001),
            "T (C°)": st.column_config.NumberColumn("T (C°)", min_value=0.0,step=0.00001),
            "Manning": st.column_config.NumberColumn("Manning", min_value=0.0,step=0.00001),
            "X-inicio (m)": st.column_config.NumberColumn("X-inicio (m)", min_value=0.0,step=0.00001),
            "X-final (m)": st.column_config.NumberColumn("X-final (m)", min_value=0.0,step=0.00001),
            "Zb-inicial (m)": st.column_config.NumberColumn("Zb-inicial (m)", min_value=0.0000,step=0.00001),
            "Zb-final (m)": st.column_config.NumberColumn("Zb-final (m)", min_value=0.0,step=0.00001),
            "Tiempo (Simulado) (Hr.)": st.column_config.NumberColumn("Tiempo (Simulado) (Hr.)", min_value=0.0,step=0.00001)
        }
    )

with st.expander("Granulometría ",expanded=True):
    
    valores_mm=[
        0.004, 0.008, 0.016, 0.032, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8,
        16, 32, 64, 128, 256, 512, 1024, 2048]
    
    valores_porcentaje=[
        0, 0, 0, 0, 3.691666667, 11.15635135, 32.43306306, 65.25578313,
        74.41965517, 77.98, 87.41115942, 95.88917722, 98.155, 100, 100,
        100, 100, 100, 100, 100]

    tamiz=pd.DataFrame([valores_porcentaje], columns=[f"{m} mm" for m in valores_mm])
    
    tamiz2=st.data_editor(
        tamiz,
        use_container_width=True,
        column_config={
        col: st.column_config.NumberColumn(col, min_value=0.0, max_value=100.0, step=0.00001)
        for col in tamiz.columns
    }
    )
    
st.title("🔘 Botón de Ejecución")

# Botón
if st.button("▶️ Ejecutar predición"):
    Q=float(edited_data["Caudal (m³/s)"].iloc[0])
    Zi=float(edited_data["Zb-inicial (m)"].iloc[0])
    Zf=float(edited_data["Zb-final (m)"].iloc[0])
    Xi=float(edited_data["X-inicio (m)"].iloc[0])
    Xf=float(edited_data["X-final (m)"].iloc[0])
    manning=float(edited_data["Manning"].iloc[0])
    TC=float(edited_data["T (C°)"].iloc[0])
    Hr=float(edited_data["Tiempo (Simulado) (Hr.)"].iloc[0])
    coordenadax=np.linspace(Xi,Xf,101)
    Z=[]
    S=(Zi-Zf)/(Xf-Xi)

    st.write(f"📐 Pendiente calculada: {S:.5f}")
    
    for i in range(101):
        Z.append(Zi-coordenadax[i]*S)
    granulometria_val1=tamiz2.values.flatten().tolist()
    S=-np.gradient(Z,coordenadax)

    entrada=np.concatenate([Z,coordenadax,S,granulometria_val1,[Q],[manning],[TC],[Hr]]).reshape(1, -1)
    
    RNA_model=load_model("RNA_MLP_EROSIÓN.keras")
    
    entrada=scaler_D_RS_I.transform(entrada)
    predct=scaler_D_RS_O.inverse_transform(RNA_model.predict(entrada))[0]
    
    prediccion=pd.DataFrame(predct)
    
    st.success("✅ Zb Originales (m):")
    st.dataframe(Z, use_container_width=True)
    
    st.success("✅ Zb predichos (m):")
    st.dataframe(prediccion, use_container_width=True)
    
    fig, ax=plt.subplots(figsize=(8, 3))
    ax.plot(coordenadax,Z,"--r")
    ax.plot(coordenadax,predct,"-b")
    ax.set_title(f"Erosión en {Hr} hr.")
    ax.grid(True)
    ax.minorticks_on()
    ax.set_xlim([min(coordenadax),max(coordenadax)])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    st.pyplot(fig)