import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT ---
st.set_page_config(layout="wide")
st.title('Simulador Interactivo de Ganadería Bovina y Emisiones')
st.markdown("""
Esta aplicación te permite modelar la dinámica de un hato de ganado bovino y sus emisiones de Gases de Efecto Invernadero (GEI).
Utiliza la barra lateral para ajustar los parámetros del modelo y observa cómo cambian las proyecciones en los gráficos.
""")

# --- BARRA LATERAL CON CONTROLES DE ENTRADA ---
st.sidebar.header('Parámetros del Escenario')

# --- Parámetros de Crecimiento ---
st.sidebar.subheader('Parámetros de Crecimiento (kg y kg/día)')
# Hembras
ganancia_peso_diario_ternera_h = st.sidebar.slider('Ganancia Peso Diario Ternera', 0.1, 1.5, 0.68, 0.01)
peso_inicial_novilla = st.sidebar.number_input('Peso Inicial Novilla', value=190)
peso_final_novilla = st.sidebar.number_input('Peso Final Novilla', value=400)
ganancia_peso_diario_novilla = st.sidebar.slider('Ganancia Peso Diario Novilla', 0.1, 1.5, 0.3, 0.01)
# Machos
peso_inicial_ternero_m = st.sidebar.number_input('Peso Inicial Ternero', value=28)
peso_final_ternero_m = st.sidebar.number_input('Peso Final Ternero', value=230)
ganancia_peso_diario_ternero_m = st.sidebar.slider('Ganancia Peso Diario Ternero', 0.1, 1.5, 0.725, 0.01)
peso_inicial_novillo_m = st.sidebar.number_input('Peso Inicial Novillo', value=230)
peso_final_novillo_m = st.sidebar.number_input('Peso Final Novillo', value=300)
ganancia_peso_diario_novillo_m = st.sidebar.slider('Ganancia Peso Diario Novillo', 0.1, 1.5, 0.25, 0.01)
peso_inicial_toro = st.sidebar.number_input('Peso Inicial Toro', value=300)
peso_final_toro = st.sidebar.number_input('Peso Final Toro', value=530)
ganancia_peso_diario_toro = st.sidebar.slider('Ganancia Peso Diario Toro', 0.1, 1.5, 0.5, 0.01)


# --- Parámetros Reproductivos y de Hato ---
st.sidebar.subheader('Parámetros Reproductivos y de Hato (%)')
porc_novillas_preñadas = st.sidebar.slider('% Novillas Preñadas', 0.0, 1.0, 0.80, 0.01)
porc_vacas_preñadas = st.sidebar.slider('% Vacas Preñadas', 0.0, 1.0, 0.85, 0.01)
porc_partos_novillas = st.sidebar.slider('% Partos Novillas', 0.0, 1.0, 0.90, 0.01)
porc_partos_vacas = st.sidebar.slider('% Partos Vacas', 0.0, 1.0, 0.95, 0.01)
porc_hembra = st.sidebar.slider('% Nacimientos Hembra', 0.0, 1.0, 0.50, 0.01)
porc_macho = 1.0 - porc_hembra # Se calcula automáticamente

# --- Parámetros de Descarte ---
st.sidebar.subheader('Parámetros de Descarte Anual (%)')
porc_descarte_novillas_no_preñadas = st.sidebar.slider('% Descarte Novillas no Preñadas', 0.0, 1.0, 0.20, 0.01)
porc_descarte_vacas = st.sidebar.slider('% Descarte Vacas', 0.0, 1.0, 0.15, 0.01)

# --- Tasas de Mortalidad ---
st.sidebar.subheader('Tasas de Mortalidad Anual (%)')
tasa_muerte_terneras = st.sidebar.slider('Tasa Muerte Terneras', 0.0, 0.2, 0.05, 0.01)
tasa_muerte_novillas = st.sidebar.slider('Tasa Muerte Novillas', 0.0, 0.2, 0.02, 0.01)
tasa_muerte_vacas = st.sidebar.slider('Tasa Muerte Vacas', 0.0, 0.2, 0.02, 0.01)
tasa_muerte_terneros = st.sidebar.slider('Tasa Muerte Terneros', 0.0, 0.2, 0.05, 0.01)
tasa_muerte_novillos = st.sidebar.slider('Tasa Muerte Novillos', 0.0, 0.2, 0.02, 0.01)
tasa_muerte_toros = st.sidebar.slider('Tasa Muerte Toros', 0.0, 0.2, 0.02, 0.01)


# --- FUNCIÓN DE SIMULACIÓN (TU CÓDIGO ADAPTADO) ---
def run_simulation(
    ganancia_peso_diario_ternera_h, peso_inicial_novilla, peso_final_novilla, ganancia_peso_diario_novilla,
    peso_inicial_ternero_m, peso_final_ternero_m, ganancia_peso_diario_ternero_m,
    peso_inicial_novillo_m, peso_final_novillo_m, ganancia_peso_diario_novillo_m,
    peso_inicial_toro, peso_final_toro, ganancia_peso_diario_toro,
    porc_novillas_preñadas, porc_vacas_preñadas, porc_partos_novillas, porc_partos_vacas,
    porc_hembra, porc_macho, porc_descarte_novillas_no_preñadas, porc_descarte_vacas,
    tasa_muerte_terneras, tasa_muerte_novillas, tasa_muerte_vacas,
    tasa_muerte_terneros, tasa_muerte_novillos, tasa_muerte_toros
):
    # --- 1. CONFIGURACIÓN DE LA SIMULACIÓN ---
    t_inicial = 0
    t_final = 3650  # 10 años en días
    dt = 1
    pasos = int((t_final - t_inicial) / dt)
    tiempo = np.linspace(t_inicial, t_final, pasos)

    # --- 2. PARÁMETROS Y CONSTANTES DEL MODELO ---
    # Parámetros fijos (no modificables por el usuario en esta versión)
    Peso_de_Venta_Novillas_Descarte = 380
    Peso_de_Venta_Vacas_Descarte = 450
    Rendimiento_Canal_Novillas = 0.53
    Rendimiento_Canal_Vacas = 0.51
    Rendimiento_Canal_Toros = 0.55
    Peso_Inicial_Ternera_H = 28
    Peso_Final_Ternera_H = 190
    Ratio_Toros_por_Vaca = 25
    Factor_Emision_Ternera = 0.5
    Factor_Emision_Novilla = 1.5
    Factor_Emision_Vaca = 2.5
    Factor_Emision_Ternero = 0.5
    Factor_Emision_Novillo = 1.8
    Factor_Emision_Toro = 2.8

    # --- 3. INICIALIZACIÓN DE VARIABLES ---
    Terneras = np.zeros(pasos)
    Novillas = np.zeros(pasos)
    Vacas = np.zeros(pasos)
    Terneros = np.zeros(pasos)
    Novillos = np.zeros(pasos)
    Toros = np.zeros(pasos)
    Carne_Producida_Acumulada = np.zeros(pasos)
    Emisiones_Acumuladas = np.zeros(pasos)
    Emisiones_Totales_GEI = np.zeros(pasos)
    Intensidad_de_Emisiones = np.zeros(pasos)

    # Población inicial
    Terneras[0] = 2000000
    Novillas[0] = 2000000
    Vacas[0] = 6800000
    Terneros[0] = 2400000
    Novillos[0] = 2000000
    Toros[0] = 1800000

    # --- 4. BUCLE DE SIMULACIÓN ---
    # Tiempos de maduración calculados a partir de los inputs
    Tiempo_Maduracion_Ternera = (Peso_Final_Ternera_H - Peso_Inicial_Ternera_H) / ganancia_peso_diario_ternera_h if ganancia_peso_diario_ternera_h > 0 else float('inf')
    Tiempo_Maduracion_Novilla = (peso_final_novilla - peso_inicial_novilla) / ganancia_peso_diario_novilla if ganancia_peso_diario_novilla > 0 else float('inf')
    Tiempo_Maduracion_Ternero_a_Novillo = (peso_final_ternero_m - peso_inicial_ternero_m) / ganancia_peso_diario_ternero_m if ganancia_peso_diario_ternero_m > 0 else float('inf')
    Tiempo_Maduracion_Novillo_a_Toro = (peso_final_novillo_m - peso_inicial_novillo_m) / ganancia_peso_diario_novillo_m if ganancia_peso_diario_novillo_m > 0 else float('inf')
    Tiempo_Engorde_Toro = (peso_final_toro - peso_inicial_toro) / ganancia_peso_diario_toro if ganancia_peso_diario_toro > 0 else float('inf')

    for i in range(1, pasos):
        # --- CÁLCULO DE FLUJOS ---
        Nacimientos_de_Novillas = Novillas[i-1] * porc_novillas_preñadas * porc_partos_novillas
        Nacimientos_de_Vacas = Vacas[i-1] * porc_vacas_preñadas * porc_partos_vacas
        Nacimientos_Totales = (Nacimientos_de_Novillas + Nacimientos_de_Vacas) / 365

        flujo_nac_hembras = Nacimientos_Totales * porc_hembra
        flujo_nac_machos = Nacimientos_Totales * porc_macho

        flujo_mad_terneras = Terneras[i-1] / Tiempo_Maduracion_Ternera if Tiempo_Maduracion_Ternera > 0 else 0
        
        novillas_no_preñadas = Novillas[i-1] * (1 - porc_novillas_preñadas)
        flujo_venta_novillas = (novillas_no_preñadas * porc_descarte_novillas_no_preñadas) / Tiempo_Maduracion_Novilla if Tiempo_Maduracion_Novilla > 0 else 0
        
        novillas_preñadas = Novillas[i-1] * porc_novillas_preñadas
        novillas_retenidas_no_preñadas = novillas_no_preñadas * (1 - porc_descarte_novillas_no_preñadas)
        flujo_conv_a_vacas = (novillas_preñadas + novillas_retenidas_no_preñadas) / Tiempo_Maduracion_Novilla if Tiempo_Maduracion_Novilla > 0 else 0
        
        flujo_venta_vacas = Vacas[i-1] * porc_descarte_vacas / 365

        toros_requeridos = Vacas[i-1] / Ratio_Toros_por_Vaca
        toros_excedentes = max(0, Toros[i-1] - toros_requeridos)
        
        flujo_mad_terneros = Terneros[i-1] / Tiempo_Maduracion_Ternero_a_Novillo if Tiempo_Maduracion_Ternero_a_Novillo > 0 else 0
        flujo_mad_novillos = Novillos[i-1] / Tiempo_Maduracion_Novillo_a_Toro if Tiempo_Maduracion_Novillo_a_Toro > 0 else 0
        flujo_venta_toros = toros_excedentes / Tiempo_Engorde_Toro if Tiempo_Engorde_Toro > 0 else 0

        flujo_mue_terneras = Terneras[i-1] * tasa_muerte_terneras / 365
        flujo_mue_novillas = Novillas[i-1] * tasa_muerte_novillas / 365
        flujo_mue_vacas = Vacas[i-1] * tasa_muerte_vacas / 365
        flujo_mue_terneros_m = Terneros[i-1] * tasa_muerte_terneros / 365
        flujo_mue_novillos_m = Novillos[i-1] * tasa_muerte_novillos / 365
        flujo_mue_toros_m = Toros[i-1] * tasa_muerte_toros / 365
        
        produccion_carne_novillas = flujo_venta_novillas * Peso_de_Venta_Novillas_Descarte * Rendimiento_Canal_Novillas
        produccion_carne_vacas = flujo_venta_vacas * Peso_de_Venta_Vacas_Descarte * Rendimiento_Canal_Vacas
        produccion_carne_toros = flujo_venta_toros * peso_final_toro * Rendimiento_Canal_Toros
        flujo_produccion_carne_total = produccion_carne_novillas + produccion_carne_vacas + produccion_carne_toros

        # --- ACTUALIZACIÓN DE NIVELES ---
        Terneras[i] = Terneras[i-1] + (flujo_nac_hembras - flujo_mad_terneras - flujo_mue_terneras) * dt
        Novillas[i] = Novillas[i-1] + (flujo_mad_terneras - flujo_conv_a_vacas - flujo_venta_novillas - flujo_mue_novillas) * dt
        Vacas[i] = Vacas[i-1] + (flujo_conv_a_vacas - flujo_venta_vacas - flujo_mue_vacas) * dt
        Terneros[i] = Terneros[i-1] + (flujo_nac_machos - flujo_mad_terneros - flujo_mue_terneros_m) * dt
        Novillos[i] = Novillos[i-1] + (flujo_mad_terneros - flujo_mad_novillos - flujo_mue_novillos_m) * dt
        Toros[i] = Toros[i-1] + (flujo_mad_novillos - flujo_venta_toros - flujo_mue_toros_m) * dt
        
        Terneras[i], Novillas[i], Vacas[i] = max(0, Terneras[i]), max(0, Novillas[i]), max(0, Vacas[i])
        Terneros[i], Novillos[i], Toros[i] = max(0, Terneros[i]), max(0, Novillos[i]), max(0, Toros[i])

        # --- CÁLCULO DE EMISIONES Y EFICIENCIA ---
        emisiones_h = (Terneras[i] * Factor_Emision_Ternera) + (Novillas[i] * Factor_Emision_Novilla) + (Vacas[i] * Factor_Emision_Vaca)
        emisiones_m = (Terneros[i] * Factor_Emision_Ternero) + (Novillos[i] * Factor_Emision_Novillo) + (Toros[i] * Factor_Emision_Toro)
        Emisiones_Totales_GEI[i] = emisiones_h + emisiones_m
        
        Carne_Producida_Acumulada[i] = Carne_Producida_Acumulada[i-1] + flujo_produccion_carne_total * dt
        Emisiones_Acumuladas[i] = Emisiones_Acumuladas[i-1] + Emisiones_Totales_GEI[i] * dt
        
        if Carne_Producida_Acumulada[i] > 0:
            Intensidad_de_Emisiones[i] = Emisiones_Acumuladas[i] / Carne_Producida_Acumulada[i]
        else:
            Intensidad_de_Emisiones[i] = 0
            
    return tiempo, Terneras, Novillas, Vacas, Terneros, Novillos, Toros, Emisiones_Totales_GEI, Intensidad_de_Emisiones

# --- EJECUCIÓN DE LA SIMULACIÓN Y VISUALIZACIÓN ---
# Llama a la función de simulación con los valores de la barra lateral
(tiempo, Terneras, Novillas, Vacas, Terneros, Novillos, Toros, 
 Emisiones_Totales_GEI, Intensidad_de_Emisiones) = run_simulation(
    ganancia_peso_diario_ternera_h, peso_inicial_novilla, peso_final_novilla, ganancia_peso_diario_novilla,
    peso_inicial_ternero_m, peso_final_ternero_m, ganancia_peso_diario_ternero_m,
    peso_inicial_novillo_m, peso_final_novillo_m, ganancia_peso_diario_novillo_m,
    peso_inicial_toro, peso_final_toro, ganancia_peso_diario_toro,
    porc_novillas_preñadas, porc_vacas_preñadas, porc_partos_novillas, porc_partos_vacas,
    porc_hembra, porc_macho, porc_descarte_novillas_no_preñadas, porc_descarte_vacas,
    tasa_muerte_terneras, tasa_muerte_novillas, tasa_muerte_vacas,
    tasa_muerte_terneros, tasa_muerte_novillos, tasa_muerte_toros
)

# --- CREACIÓN DE GRÁFICOS ---
st.header('Resultados de la Simulación')
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20), sharex=True)
tiempo_en_años = tiempo / 365

# Gráfico 1: Población de Hembras
ax1.plot(tiempo_en_años, Terneras, label='Terneras', color='deeppink')
ax1.plot(tiempo_en_años, Novillas, label='Novillas', color='mediumvioletred')
ax1.plot(tiempo_en_años, Vacas, label='Vacas', color='darkmagenta')
ax1.set_title('Evolución de la Población de Hembras', fontsize=14)
ax1.set_ylabel('Número de Animales')
ax1.legend()
ax1.grid(True)

# Gráfico 2: Población de Machos
ax2.plot(tiempo_en_años, Terneros, label='Terneros', color='dodgerblue')
ax2.plot(tiempo_en_años, Novillos, label='Novillos', color='royalblue')
ax2.plot(tiempo_en_años, Toros, label='Toros', color='navy')
ax2.set_title('Evolución de la Población de Machos', fontsize=14)
ax2.set_ylabel('Número de Animales')
ax2.legend()
ax2.grid(True)

# Gráfico 3: Emisiones Totales de GEI
ax3.plot(tiempo_en_años[1:], Emisiones_Totales_GEI[1:], label='Emisiones Totales GEI', color='forestgreen')
ax3.set_title('Evolución de las Emisiones Totales de GEI', fontsize=14)
ax3.set_ylabel('Emisiones GEI (kg CO2-eq/día)')
ax3.legend()
ax3.grid(True)

# Gráfico 4: Intensidad de Emisiones
ax4.plot(tiempo_en_años[1:], Intensidad_de_Emisiones[1:], label='Intensidad de Emisiones', color='firebrick')
ax4.set_title('Evolución de la Intensidad de Emisiones', fontsize=14)
ax4.set_xlabel('Tiempo (Años)')
ax4.set_ylabel('kg CO2-eq / kg Carne')
ax4.legend()
ax4.grid(True)
if np.max(Intensidad_de_Emisiones) > 0:
    ax4.set_ylim(bottom=0)

plt.tight_layout()

# Mostrar el gráfico en Streamlit
st.pyplot(fig)

