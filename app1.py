import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT ---
st.set_page_config(layout="wide")
st.title('Simulador Comparativo de Escenarios de Ganadería Bovina')
st.markdown("""
Esta herramienta permite comparar un escenario **Línea Base** con un **Escenario Alternativo**. 
Modifica los parámetros en la barra lateral para configurar el Escenario Alternativo y observa el impacto en los indicadores clave y las proyecciones a 10 años.
""")

# --- BARRA LATERAL: CONTROLES PARA EL ESCENARIO ALTERNATIVO ---
st.sidebar.header('Parámetros del Escenario Alternativo')

# --- Parámetros de Crecimiento ---
st.sidebar.subheader('Parámetros de Crecimiento (kg y kg/día)')
alt_ganancia_peso_diario_ternera_h = st.sidebar.slider('Ganancia Peso Diario Ternera', 0.1, 1.5, 0.68, 0.01, key='alt_gpdt')
alt_peso_inicial_novilla = st.sidebar.number_input('Peso Inicial Novilla', value=190, key='alt_pin')
alt_peso_final_novilla = st.sidebar.number_input('Peso Final Novilla', value=400, key='alt_pfn')
alt_ganancia_peso_diario_novilla = st.sidebar.slider('Ganancia Peso Diario Novilla', 0.1, 1.5, 0.3, 0.01, key='alt_gpdn')
alt_peso_inicial_ternero_m = st.sidebar.number_input('Peso Inicial Ternero', value=28, key='alt_pit')
alt_peso_final_ternero_m = st.sidebar.number_input('Peso Final Ternero', value=230, key='alt_pft')
alt_ganancia_peso_diario_ternero_m = st.sidebar.slider('Ganancia Peso Diario Ternero', 0.1, 1.5, 0.725, 0.01, key='alt_gpdtm')
alt_peso_inicial_novillo_m = st.sidebar.number_input('Peso Inicial Novillo', value=230, key='alt_pinm')
alt_peso_final_novillo_m = st.sidebar.number_input('Peso Final Novillo', value=300, key='alt_pfnm')
alt_ganancia_peso_diario_novillo_m = st.sidebar.slider('Ganancia Peso Diario Novillo', 0.1, 1.5, 0.25, 0.01, key='alt_gpdnm')
alt_peso_inicial_toro = st.sidebar.number_input('Peso Inicial Toro', value=300, key='alt_pito')
alt_peso_final_toro = st.sidebar.number_input('Peso Final Toro', value=530, key='alt_pfto')
alt_ganancia_peso_diario_toro = st.sidebar.slider('Ganancia Peso Diario Toro', 0.1, 1.5, 0.5, 0.01, key='alt_gpdto')

# --- Parámetros Reproductivos y de Hato ---
st.sidebar.subheader('Parámetros Reproductivos y de Hato (%)')
alt_porc_novillas_preñadas = st.sidebar.slider('% Novillas Preñadas', 0.0, 1.0, 0.80, 0.01, key='alt_pnp')
alt_porc_vacas_preñadas = st.sidebar.slider('% Vacas Preñadas', 0.0, 1.0, 0.85, 0.01, key='alt_pvp')
alt_porc_partos_novillas = st.sidebar.slider('% Partos Novillas', 0.0, 1.0, 0.90, 0.01, key='alt_ppn')
alt_porc_partos_vacas = st.sidebar.slider('% Partos Vacas', 0.0, 1.0, 0.95, 0.01, key='alt_ppv')
alt_porc_hembra = st.sidebar.slider('% Nacimientos Hembra', 0.0, 1.0, 0.50, 0.01, key='alt_ph')

# --- Parámetros de Descarte ---
st.sidebar.subheader('Parámetros de Descarte Anual (%)')
alt_porc_descarte_novillas_no_preñadas = st.sidebar.slider('% Descarte Novillas no Preñadas', 0.0, 1.0, 0.20, 0.01, key='alt_pdnnp')
alt_porc_descarte_vacas = st.sidebar.slider('% Descarte Vacas', 0.0, 1.0, 0.15, 0.01, key='alt_pdv')

# --- Tasas de Mortalidad ---
st.sidebar.subheader('Tasas de Mortalidad Anual (%)')
alt_tasa_muerte_terneras = st.sidebar.slider('Tasa Muerte Terneras', 0.0, 0.2, 0.05, 0.01, key='alt_tmt')
alt_tasa_muerte_novillas = st.sidebar.slider('Tasa Muerte Novillas', 0.0, 0.2, 0.02, 0.01, key='alt_tmn')
alt_tasa_muerte_vacas = st.sidebar.slider('Tasa Muerte Vacas', 0.0, 0.2, 0.02, 0.01, key='alt_tmv')
alt_tasa_muerte_terneros = st.sidebar.slider('Tasa Muerte Terneros', 0.0, 0.2, 0.05, 0.01, key='alt_tmtm')
alt_tasa_muerte_novillos = st.sidebar.slider('Tasa Muerte Novillos', 0.0, 0.2, 0.02, 0.01, key='alt_tmnm')
alt_tasa_muerte_toros = st.sidebar.slider('Tasa Muerte Toros', 0.0, 0.2, 0.02, 0.01, key='alt_tmtoro')


# --- FUNCIÓN DE SIMULACIÓN (Reutilizable para ambos escenarios) ---
@st.cache_data
def run_simulation(params):
    t_inicial, t_final, dt = 0, 3650, 1
    pasos = int((t_final - t_inicial) / dt)
    tiempo = np.linspace(t_inicial, t_final, pasos)
    
    # Inicialización de arrays
    Terneras, Novillas, Vacas = np.zeros(pasos), np.zeros(pasos), np.zeros(pasos)
    Terneros, Novillos, Toros = np.zeros(pasos), np.zeros(pasos), np.zeros(pasos)
    Carne_Producida_Acumulada, Emisiones_Acumuladas = np.zeros(pasos), np.zeros(pasos)
    Emisiones_Totales_GEI, Intensidad_de_Emisiones = np.zeros(pasos), np.zeros(pasos)

    # Población inicial
    Terneras[0], Novillas[0], Vacas[0] = 2000000, 2000000, 6800000
    Terneros[0], Novillos[0], Toros[0] = 2400000, 2000000, 1800000

    # Parámetros fijos
    Peso_de_Venta_Novillas_Descarte, Peso_de_Venta_Vacas_Descarte = 380, 450
    Rendimiento_Canal_Novillas, Rendimiento_Canal_Vacas, Rendimiento_Canal_Toros = 0.53, 0.51, 0.55
    Peso_Inicial_Ternera_H, Peso_Final_Ternera_H = 28, 190
    Ratio_Toros_por_Vaca = 25
    Factor_Emision_Ternera, Factor_Emision_Novilla, Factor_Emision_Vaca = 0.5, 1.5, 2.5
    Factor_Emision_Ternero, Factor_Emision_Novillo, Factor_Emision_Toro = 0.5, 1.8, 2.8
    
    # Tiempos de maduración
    Tiempo_Maduracion_Ternera = (Peso_Final_Ternera_H - Peso_Inicial_Ternera_H) / params['ganancia_peso_diario_ternera_h'] if params['ganancia_peso_diario_ternera_h'] > 0 else float('inf')
    Tiempo_Maduracion_Novilla = (params['peso_final_novilla'] - params['peso_inicial_novilla']) / params['ganancia_peso_diario_novilla'] if params['ganancia_peso_diario_novilla'] > 0 else float('inf')
    Tiempo_Maduracion_Ternero_a_Novillo = (params['peso_final_ternero_m'] - params['peso_inicial_ternero_m']) / params['ganancia_peso_diario_ternero_m'] if params['ganancia_peso_diario_ternero_m'] > 0 else float('inf')
    Tiempo_Maduracion_Novillo_a_Toro = (params['peso_final_novillo_m'] - params['peso_inicial_novillo_m']) / params['ganancia_peso_diario_novillo_m'] if params['ganancia_peso_diario_novillo_m'] > 0 else float('inf')
    Tiempo_Engorde_Toro = (params['peso_final_toro'] - params['peso_inicial_toro']) / params['ganancia_peso_diario_toro'] if params['ganancia_peso_diario_toro'] > 0 else float('inf')

    for i in range(1, pasos):
        # Lógica de simulación (sin cambios)
        Nacimientos_de_Novillas = Novillas[i-1] * params['porc_novillas_preñadas'] * params['porc_partos_novillas']
        Nacimientos_de_Vacas = Vacas[i-1] * params['porc_vacas_preñadas'] * params['porc_partos_vacas']
        Nacimientos_Totales = (Nacimientos_de_Novillas + Nacimientos_de_Vacas) / 365
        flujo_nac_hembras = Nacimientos_Totales * params['porc_hembra']
        flujo_nac_machos = Nacimientos_Totales * (1.0 - params['porc_hembra'])
        flujo_mad_terneras = Terneras[i-1] / Tiempo_Maduracion_Ternera if Tiempo_Maduracion_Ternera > 0 else 0
        novillas_no_preñadas = Novillas[i-1] * (1 - params['porc_novillas_preñadas'])
        flujo_venta_novillas = (novillas_no_preñadas * params['porc_descarte_novillas_no_preñadas']) / Tiempo_Maduracion_Novilla if Tiempo_Maduracion_Novilla > 0 else 0
        novillas_preñadas = Novillas[i-1] * params['porc_novillas_preñadas']
        novillas_retenidas_no_preñadas = novillas_no_preñadas * (1 - params['porc_descarte_novillas_no_preñadas'])
        flujo_conv_a_vacas = (novillas_preñadas + novillas_retenidas_no_preñadas) / Tiempo_Maduracion_Novilla if Tiempo_Maduracion_Novilla > 0 else 0
        flujo_venta_vacas = Vacas[i-1] * params['porc_descarte_vacas'] / 365
        toros_requeridos = Vacas[i-1] / Ratio_Toros_por_Vaca
        toros_excedentes = max(0, Toros[i-1] - toros_requeridos)
        flujo_mad_terneros = Terneros[i-1] / Tiempo_Maduracion_Ternero_a_Novillo if Tiempo_Maduracion_Ternero_a_Novillo > 0 else 0
        flujo_mad_novillos = Novillos[i-1] / Tiempo_Maduracion_Novillo_a_Toro if Tiempo_Maduracion_Novillo_a_Toro > 0 else 0
        flujo_venta_toros = toros_excedentes / Tiempo_Engorde_Toro if Tiempo_Engorde_Toro > 0 else 0
        flujo_mue_terneras = Terneras[i-1] * params['tasa_muerte_terneras'] / 365
        flujo_mue_novillas = Novillas[i-1] * params['tasa_muerte_novillas'] / 365
        flujo_mue_vacas = Vacas[i-1] * params['tasa_muerte_vacas'] / 365
        flujo_mue_terneros_m = Terneros[i-1] * params['tasa_muerte_terneros'] / 365
        flujo_mue_novillos_m = Novillos[i-1] * params['tasa_muerte_novillos'] / 365
        flujo_mue_toros_m = Toros[i-1] * params['tasa_muerte_toros'] / 365
        produccion_carne_novillas = flujo_venta_novillas * Peso_de_Venta_Novillas_Descarte * Rendimiento_Canal_Novillas
        produccion_carne_vacas = flujo_venta_vacas * Peso_de_Venta_Vacas_Descarte * Rendimiento_Canal_Vacas
        produccion_carne_toros = flujo_venta_toros * params['peso_final_toro'] * Rendimiento_Canal_Toros
        flujo_produccion_carne_total = produccion_carne_novillas + produccion_carne_vacas + produccion_carne_toros
        
        Terneras[i] = Terneras[i-1] + (flujo_nac_hembras - flujo_mad_terneras - flujo_mue_terneras) * dt
        Novillas[i] = Novillas[i-1] + (flujo_mad_terneras - flujo_conv_a_vacas - flujo_venta_novillas - flujo_mue_novillas) * dt
        Vacas[i] = Vacas[i-1] + (flujo_conv_a_vacas - flujo_venta_vacas - flujo_mue_vacas) * dt
        Terneros[i] = Terneros[i-1] + (flujo_nac_machos - flujo_mad_terneros - flujo_mue_terneros_m) * dt
        Novillos[i] = Novillos[i-1] + (flujo_mad_terneros - flujo_mad_novillos - flujo_mue_novillos_m) * dt
        Toros[i] = Toros[i-1] + (flujo_mad_novillos - flujo_venta_toros - flujo_mue_toros_m) * dt
        
        Terneras[i], Novillas[i], Vacas[i] = max(0, Terneras[i]), max(0, Novillas[i]), max(0, Vacas[i])
        Terneros[i], Novillos[i], Toros[i] = max(0, Terneros[i]), max(0, Novillos[i]), max(0, Toros[i])
        
        emisiones_h = (Terneras[i] * Factor_Emision_Ternera) + (Novillas[i] * Factor_Emision_Novilla) + (Vacas[i] * Factor_Emision_Vaca)
        emisiones_m = (Terneros[i] * Factor_Emision_Ternero) + (Novillos[i] * Factor_Emision_Novillo) + (Toros[i] * Factor_Emision_Toro)
        Emisiones_Totales_GEI[i] = emisiones_h + emisiones_m
        Carne_Producida_Acumulada[i] = Carne_Producida_Acumulada[i-1] + flujo_produccion_carne_total * dt
        Emisiones_Acumuladas[i] = Emisiones_Acumuladas[i-1] + Emisiones_Totales_GEI[i] * dt
        if Carne_Producida_Acumulada[i] > 0:
            Intensidad_de_Emisiones[i] = Emisiones_Acumuladas[i] / Carne_Producida_Acumulada[i]
        else:
            Intensidad_de_Emisiones[i] = 0
            
    return {
        "tiempo": tiempo, "Terneras": Terneras, "Novillas": Novillas, "Vacas": Vacas,
        "Terneros": Terneros, "Novillos": Novillos, "Toros": Toros,
        "Emisiones_Totales_GEI": Emisiones_Totales_GEI,
        "Intensidad_de_Emisiones": Intensidad_de_Emisiones
    }

# --- DEFINICIÓN DE PARÁMETROS PARA AMBOS ESCENARIOS ---
base_params = {
    'ganancia_peso_diario_ternera_h': 0.68, 'peso_inicial_novilla': 190, 'peso_final_novilla': 400,
    'ganancia_peso_diario_novilla': 0.3, 'peso_inicial_ternero_m': 28, 'peso_final_ternero_m': 230,
    'ganancia_peso_diario_ternero_m': 0.725, 'peso_inicial_novillo_m': 230, 'peso_final_novillo_m': 300,
    'ganancia_peso_diario_novillo_m': 0.25, 'peso_inicial_toro': 300, 'peso_final_toro': 530,
    'ganancia_peso_diario_toro': 0.5, 'porc_novillas_preñadas': 0.80, 'porc_vacas_preñadas': 0.85,
    'porc_partos_novillas': 0.90, 'porc_partos_vacas': 0.95, 'porc_hembra': 0.50,
    'porc_descarte_novillas_no_preñadas': 0.20, 'porc_descarte_vacas': 0.15,
    'tasa_muerte_terneras': 0.05, 'tasa_muerte_novillas': 0.02, 'tasa_muerte_vacas': 0.02,
    'tasa_muerte_terneros': 0.05, 'tasa_muerte_novillos': 0.02, 'tasa_muerte_toros': 0.02
}

alt_params = {
    'ganancia_peso_diario_ternera_h': alt_ganancia_peso_diario_ternera_h, 'peso_inicial_novilla': alt_peso_inicial_novilla,
    'peso_final_novilla': alt_peso_final_novilla, 'ganancia_peso_diario_novilla': alt_ganancia_peso_diario_novilla,
    'peso_inicial_ternero_m': alt_peso_inicial_ternero_m, 'peso_final_ternero_m': alt_peso_final_ternero_m,
    'ganancia_peso_diario_ternero_m': alt_ganancia_peso_diario_ternero_m, 'peso_inicial_novillo_m': alt_peso_inicial_novillo_m,
    'peso_final_novillo_m': alt_peso_final_novillo_m, 'ganancia_peso_diario_novillo_m': alt_ganancia_peso_diario_novillo_m,
    'peso_inicial_toro': alt_peso_inicial_toro, 'peso_final_toro': alt_peso_final_toro,
    'ganancia_peso_diario_toro': alt_ganancia_peso_diario_toro, 'porc_novillas_preñadas': alt_porc_novillas_preñadas,
    'porc_vacas_preñadas': alt_porc_vacas_preñadas, 'porc_partos_novillas': alt_porc_partos_novillas,
    'porc_partos_vacas': alt_porc_partos_vacas, 'porc_hembra': alt_porc_hembra,
    'porc_descarte_novillas_no_preñadas': alt_porc_descarte_novillas_no_preñadas, 'porc_descarte_vacas': alt_porc_descarte_vacas,
    'tasa_muerte_terneras': alt_tasa_muerte_terneras, 'tasa_muerte_novillas': alt_tasa_muerte_novillas,
    'tasa_muerte_vacas': alt_tasa_muerte_vacas, 'tasa_muerte_terneros': alt_tasa_muerte_terneros,
    'tasa_muerte_novillos': alt_tasa_muerte_novillos, 'tasa_muerte_toros': alt_tasa_muerte_toros
}

# --- EJECUCIÓN DE SIMULACIONES ---
base_results = run_simulation(base_params)
alt_results = run_simulation(alt_params)

# --- CÁLCULO Y VISUALIZACIÓN DE KPIs ---
st.header('Indicadores Clave de Rendimiento (KPIs) al Año 10')

base_emisiones_final = base_results['Emisiones_Totales_GEI'][-1]
alt_emisiones_final = alt_results['Emisiones_Totales_GEI'][-1]
delta_emisiones = ((alt_emisiones_final - base_emisiones_final) / base_emisiones_final) * 100 if base_emisiones_final != 0 else 0

base_intensidad_final = base_results['Intensidad_de_Emisiones'][-1]
alt_intensidad_final = alt_results['Intensidad_de_Emisiones'][-1]
delta_intensidad = ((alt_intensidad_final - base_intensidad_final) / base_intensidad_final) * 100 if base_intensidad_final != 0 else 0

col1, col2 = st.columns(2)
with col1:
    st.metric(
        label="Emisiones Totales Diarias (kg CO2-eq)",
        value=f"{alt_emisiones_final:,.0f}",
        delta=f"{delta_emisiones:.2f}% vs Línea Base",
        delta_color="inverse"
    )
with col2:
    st.metric(
        label="Intensidad de Emisiones (kg CO2-eq / kg Carne)",
        value=f"{alt_intensidad_final:.2f}",
        delta=f"{delta_intensidad:.2f}% vs Línea Base",
        delta_color="inverse"
    )

st.markdown("---")
st.header('Visualización Comparativa de Escenarios')

# --- CREACIÓN DE GRÁFICOS COMPARATIVOS ---
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.08,
    subplot_titles=('Población de Hembras', 'Población de Machos', 'Emisiones Totales de GEI', 'Intensidad de Emisiones')
)
tiempo_en_años = base_results['tiempo'] / 365

# Función auxiliar para añadir trazas
def add_traces(fig, row, col, data_base, data_alt, name):
    fig.add_trace(go.Scatter(x=tiempo_en_años, y=data_base, name=f'{name} (Base)', mode='lines', line=dict(color=f'rgba({np.random.randint(0,255)},{np.random.randint(0,255)},{np.random.randint(0,255)},0.6)', width=2)), row=row, col=col)
    fig.add_trace(go.Scatter(x=tiempo_en_años, y=data_alt, name=f'{name} (Alt)', mode='lines', line=dict(color=f'rgba({np.random.randint(0,255)},{np.random.randint(0,255)},{np.random.randint(0,255)},1)', width=3, dash='dash')), row=row, col=col)

# Gráfico 1: Hembras
add_traces(fig, 1, 1, base_results['Terneras'], alt_results['Terneras'], 'Terneras')
add_traces(fig, 1, 1, base_results['Novillas'], alt_results['Novillas'], 'Novillas')
add_traces(fig, 1, 1, base_results['Vacas'], alt_results['Vacas'], 'Vacas')

# Gráfico 2: Machos
add_traces(fig, 2, 1, base_results['Terneros'], alt_results['Terneros'], 'Terneros')
add_traces(fig, 2, 1, base_results['Novillos'], alt_results['Novillos'], 'Novillos')
add_traces(fig, 2, 1, base_results['Toros'], alt_results['Toros'], 'Toros')

# Gráfico 3: Emisiones
fig.add_trace(go.Scatter(x=tiempo_en_años, y=base_results['Emisiones_Totales_GEI'], name='Emisiones GEI (Base)', mode='lines', line=dict(color='grey')), row=3, col=1)
fig.add_trace(go.Scatter(x=tiempo_en_años, y=alt_results['Emisiones_Totales_GEI'], name='Emisiones GEI (Alt)', mode='lines', line=dict(color='green', dash='dash')), row=3, col=1)

# Gráfico 4: Intensidad
fig.add_trace(go.Scatter(x=tiempo_en_años, y=base_results['Intensidad_de_Emisiones'], name='Intensidad (Base)', mode='lines', line=dict(color='grey')), row=4, col=1)
fig.add_trace(go.Scatter(x=tiempo_en_años, y=alt_results['Intensidad_de_Emisiones'], name='Intensidad (Alt)', mode='lines', line=dict(color='firebrick', dash='dash')), row=4, col=1)

# Diseño de la figura
fig.update_layout(height=1000, template='plotly_white', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig.update_yaxes(title_text="Número de Animales", row=1, col=1); fig.update_yaxes(title_text="Número de Animales", row=2, col=1)
fig.update_yaxes(title_text="kg CO2-eq/día", row=3, col=1); fig.update_yaxes(title_text="kg CO2-eq / kg Carne", row=4, col=1)
fig.update_xaxes(title_text="Tiempo (Años)", row=4, col=1)
st.plotly_chart(fig, use_container_width=True)

# --- DESCARGA DE DATOS ---
st.markdown("---")
st.header('Descargar Datos de la Simulación')

# Preparar DataFrame para descarga
df_base = pd.DataFrame(base_results)
df_alt = pd.DataFrame(alt_results)
df_base = df_base.add_suffix('_Base')
df_alt = df_alt.add_suffix('_Alternativo').drop(columns=['tiempo_Alternativo'])
df_export = pd.concat([df_base, df_alt], axis=1)
df_export['Año'] = df_export['tiempo_Base'] / 365
df_export = df_export.set_index('Año')

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

csv = convert_df_to_csv(df_export)

st.download_button(
    label="Descargar datos como CSV",
    data=csv,
    file_name='resultados_simulacion_ganaderia.csv',
    mime='text/csv',
)
