import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from io import BytesIO
import io, base64

#OJO E UNA DEMO

st.set_page_config(
    page_title="RNA_MLP_EROSION",
    page_icon="🌊",
    layout="wide"
)

st.markdown("""
<div style='text-align: center;'>
    <h1>🌊 RNA MLP PARA LA PREDICCIÓN DE EROSIÓN Y SEDIMENTACIÓN EN UN CANAL VARIABLE RECTANGULAR</h1>
</div>
""", unsafe_allow_html=True)


with st.expander("🔬 Descripción del proyecto",expanded=True):
    st.markdown("""
    <div style='text-align: justify; font-size: 16px;'>
    Este proyecto desarrolla una <b>herramienta web interactiva</b> basada en Python y Streamlit, diseñada para procesar las 
    <b>variables hidráulicas y geométricas</b> que sirven como entrada para un modelo de <b>red neuronal multicapa (MLP)</b>. 
    El objetivo principal es permitir la <b>predicción del comportamiento erosivo</b> en un canal rectangular a partir de 
    condiciones definidas por el usuario.

    El aplicativo integra las características esenciales utilizadas por el modelo, tales como la <b>granulometría del material</b>, 
    el <b>coeficiente de Manning</b>, el <b>caudal de flujo</b>, el <b>espaciamiento entre nodos</b>, la <b>longitud del tramo</b>, 
    la <b>temperatura del agua</b> y el <b>ancho del canal</b>. Estas variables conforman el vector de entrada con el cual la red 
    neuronal genera la predicción del perfil erosivo.

    A través de esta plataforma, el usuario puede ingresar parámetros personalizados, ejecutar el modelo entrenado y visualizar 
    los resultados de forma inmediata mediante gráficos dinámicos y métricas complementarias. Esta solución constituye un método 
    accesible, rápido y eficiente para el análisis de erosión, integrando <b>inteligencia artificial</b> dentro de un entorno 
    intuitivo orientado a la práctica ingenieril.
    </div>
    """, unsafe_allow_html=True)

with st.expander("🔬 Metricas de Entrenamiento",expanded=True):
    st.markdown("""
    <div style='text-align: center; font-size: 18px;'>
    📊 <b>MSE</b> = 0.0533<br>
    📉 <b>RMSE</b> = 0.2309<br>
    ✅ <b>MAE</b> = 0.0233<br>
    📌 <b>MAPE</b> = 17.3013 %<br>
    📈 <b>R²</b> = 0.8749<br>
    🧭 <b>MBE</b> = -0.0233
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔎 Parametros",expanded=True):
        #Crear un DataFrame de ejemplo
        data=pd.DataFrame({
            "Caudal (m³/s)": [3],
            "T (C°)": [20.0],
            "Manning":[0.035],
            "Longitud (m)":[20.0],
            "Zb-inicial (m)":[29.9999],
            "Zb-final (m)":[30.1999],
            "Ancho (m)":[3.000]
        })

        #Mostrar la tabla editable /num_rows="dynamic",
        edited_data=st.data_editor(
                data,
                use_container_width=True,
                column_config={
                "Caudal (m³/s)": st.column_config.NumberColumn(
                    "Caudal (m³/s)", min_value=0.0, step=0.00001
                ),
                "T (C°)": st.column_config.NumberColumn(
                    "T (C°)", min_value=0.0, step=0.00001
                ),
                "Manning": st.column_config.NumberColumn(
                    "Manning", min_value=0.0, step=0.00001
                ),
                "Longitud (m)": st.column_config.NumberColumn(
                    "Longitud (m)", min_value=0.0, step=0.00001
                ),
                "Zb-inicial (m)": st.column_config.NumberColumn(
                    "Zb-inicial (m)", min_value=0.0, step=0.00001
                ),
                "Zb-final (m)": st.column_config.NumberColumn(
                    "Zb-final (m)", min_value=0.0, step=0.00001
                ),
                "Ancho (m)": st.column_config.NumberColumn(
                    "Ancho (m)", min_value=0.0, step=0.00001
                )
                }
                )

    with st.expander("Granulometría ",expanded=True):
        
        valores_mm=[
            0.004, 0.008, 0.016, 0.032, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8,
            16, 32, 64, 128, 256, 512, 1024, 2048]
        
        valores_porcentaje=[
        0,
        0,
        0,
        0,
        23.3,
        38.18972973,
        59.96972973,
        76.54831325,
        84.52724138,
        88.26,
        92.24550725,
        96.39164557,
        99.24526316,
        100,
        100,
        100,
        100,
        100,
        100,
        100
    ]

        tamiz=pd.DataFrame([valores_porcentaje], columns=[f"{m} mm" for m in valores_mm])
        
        tamiz2=st.data_editor(
            tamiz,
            use_container_width=True,
            column_config={
            col: st.column_config.NumberColumn(col, min_value=0.0, max_value=100.0, step=0.00001)
            for col in tamiz.columns
        }
        )

try:        
    st.title("🔘 Botón de Ejecución")

    # Botón
    if st.button("▶️ Ejecutar predición"):
        Q=float(edited_data["Caudal (m³/s)"].iloc[0])
        Zi=float(edited_data["Zb-inicial (m)"].iloc[0])
        Zf=float(edited_data["Zb-final (m)"].iloc[0])
        manning=float(edited_data["Manning"].iloc[0])
        TC=float(edited_data["T (C°)"].iloc[0])
        long=float(edited_data["Longitud (m)"].iloc[0])
        ancho=float(edited_data["Ancho (m)"].iloc[0])
        
        coordenadax=np.linspace(0,long,101)
        z_elev=np.linspace(Zi, Zf, 101)
        S=(Zf-Zi)/long

        st.write(f"📐 Pendiente calculada: {S:.5f}")

        granulometria_val1=tamiz2.values.tolist()[0]

        entrada=np.concatenate([[manning],[Q],[ancho],[long],[TC], granulometria_val1,  z_elev]).reshape(1, -1)
        
        
        RNA_model=load_model("RNA_MLP_EROSION.h5",compile=False)
        
        predct=RNA_model.predict(entrada)[0]
        
        print(predct)
        
        prediccion=pd.DataFrame(predct)
        
        st.success("✅ Zb Originales (m):")
        st.dataframe(z_elev, use_container_width=True)
        
        st.success("✅ Zb predichos (m):")
        st.dataframe(prediccion, use_container_width=True)
        
        plt.style.use('classic')
        fig, ax=plt.subplots(figsize=(9, 3.6))
        #CURVA: Perfil inicial
        ax.plot(
            coordenadax,
            z_elev,
            linestyle="--",
            color="#d62728",         #rojo elegante
            linewidth=2,
            label="Perfil inicial"
        )

        # ---- CURVA: Predicción ----
        ax.plot(
            coordenadax,
            predct,
            linestyle="-",
            color="#1f77b4",         #azul clásico
            linewidth=2.3,
            label="Predicción"
        )

        #TÍTULO Y ETIQUETAS
        ax.set_title("Erosión y sedimentación", fontsize=13, fontweight="bold")
        ax.set_xlabel("X (m)", fontsize=11)
        ax.set_ylabel("Z (m)", fontsize=11)

        #RANGOS
        ax.set_xlim(min(coordenadax), max(coordenadax))

        #GRILLA
        ax.grid(True, which="both", alpha=0.28)
        ax.minorticks_on()
        ax.ticklabel_format(style='plain', axis='y')
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
    

        #LEYENDA
        ax.legend(
            fontsize=10,
            frameon=True,
            framealpha=0.90,
            edgecolor="gray",
            loc="best"
        )

        buff=io.BytesIO()
        fig.savefig(buff, format="png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
        buff.seek(0)

        b64=base64.b64encode(buff.getvalue()).decode()

        st.markdown(
            f"""
            <div style="text-align:center;">
                <img src="data:image/png;base64,{b64}" width="1100">
            </div>
            """,
            unsafe_allow_html=True
        )

except:

    print("fallo algo")


