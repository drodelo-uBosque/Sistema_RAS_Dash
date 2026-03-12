# ============================================================
# DASHBOARD COMPLETO - RAS TILAPIA
# Versión profesional con login y gráficas avanzadas
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
import os
from datetime import datetime
import streamlit_authenticator as stauth
import bcrypt  
from PIL import Image

# ── CONFIGURACIÓN ─────────────────────────────────────────────
logo_icono = Image.open('logo.png') 

st.set_page_config(                      # ← luego aquí
    page_title="RAS Tilapia - Monitor",
    page_icon=logo_icono,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── ESTILOS ───────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo general */
    .stApp { background-color: #0A1628; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D2137 0%, #0A1628 100%);
        border-right: 1px solid #1E3A5F;
    }

    /* Tarjetas métricas */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #0D2137, #1A3A5C);
        border: 1px solid #1E3A5F;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] {
        background: #0D2137;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #7FB3D3;
        border-radius: 8px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1565C0, #0D47A1) !important;
        color: white !important;
    }

    /* Alertas */
    .alerta-verde {
        background: linear-gradient(135deg, #1B5E20, #2E7D32);
        border-left: 5px solid #4CAF50;
        border-radius: 10px; padding: 15px;
        text-align: center; color: white;
        font-size: 16px; font-weight: bold;
        box-shadow: 0 4px 15px rgba(76,175,80,0.3);
    }
    .alerta-amarilla {
        background: linear-gradient(135deg, #E65100, #F57C00);
        border-left: 5px solid #FF9800;
        border-radius: 10px; padding: 15px;
        text-align: center; color: white;
        font-size: 16px; font-weight: bold;
        box-shadow: 0 4px 15px rgba(255,152,0,0.3);
    }
    .alerta-roja {
        background: linear-gradient(135deg, #B71C1C, #C62828);
        border-left: 5px solid #F44336;
        border-radius: 10px; padding: 15px;
        text-align: center; color: white;
        font-size: 16px; font-weight: bold;
        box-shadow: 0 4px 15px rgba(244,67,54,0.3);
    }

    /* Títulos */
    h1, h2, h3 { color: #E3F2FD !important; }

    /* Separadores */
    hr { border-color: #1E3A5F; }

    /* Tarjeta tanque */
    .tanque-card {
        background: linear-gradient(135deg, #0D2137, #1A3A5C);
        border: 1px solid #1E3A5F;
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        margin-bottom: 8px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ── SISTEMA DE LOGIN ──────────────────────────────────────────
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

credentials = {
    "usernames": {
        "admin": {
            "name": "Administrador",
            "password": hash_password("admin123")
        },
        "investigador": {
            "name": "Investigador RAS",
            "password": hash_password("ras2024")
        },
        "estudiante": {
            "name": "Estudiante",
            "password": hash_password("udca2024")
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "ras_tilapia_dashboard",
    "ras_key_2024",
    cookie_expiry_days=1
)

authenticator.login(location="main")
name               = st.session_state.get("name", None)
authentication_status = st.session_state.get("authentication_status", None)
username           = st.session_state.get("username", None)

# ── PANTALLA DE LOGIN ─────────────────────────────────────────
if authentication_status is False:
    st.error("❌ Usuario o contraseña incorrectos")
    st.stop()

elif authentication_status is None:
    st.markdown("""
        <div style='text-align:center; padding: 40px;'>
            <h1 style='color:#E3F2FD'>💧 RAS Tilapia Monitor</h1>
            <p style='color:#7FB3D3'>
                Sistema de monitoreo predictivo para acuicultura
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── DASHBOARD (solo si está autenticado) ─────────────────────
elif authentication_status:

    # ── CARGAR DATOS ──────────────────────────────────────────
    @st.cache_resource
    def cargar_modelo():
        with open('datos/mejor_modelo_clasificacion.pkl', 'rb') as f:
            modelo = pickle.load(f)
        with open('datos/scaler_clasificacion.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return modelo, scaler

    @st.cache_data
    def cargar_datos():
        df = pd.read_csv('dataset_udca.csv')
        df['Fecha_Hora'] = pd.to_datetime(df['Fecha_Hora'])
        return df

    @st.cache_data
    def cargar_resultados():
        with open('datos/resultados_modelos.json', 'r') as f:
            return json.load(f)

    modelo, scaler = cargar_modelo()
    df             = cargar_datos()
    resultados     = cargar_resultados()

    # ── SIDEBAR ───────────────────────────────────────────────
    with st.sidebar:
        # Logo universidad
        logo_path = 'logo.png'
        if os.path.exists(logo_path):
            st.image(logo_path, use_column_width=True)
        st.markdown("---")

        st.markdown(f"""
            <div style='text-align:center; padding:10px;'>
                <h3 style='color:#E3F2FD; margin:0'>💧 RAS Tilapia</h3>
                <p style='color:#7FB3D3; font-size:12px; margin:0'>
                    Sistema de Monitoreo Predictivo
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        st.markdown(f"👤 **{name}**")
        authenticator.logout("🚪 Cerrar sesión", "sidebar")
        st.markdown("---")

        st.subheader("🔍 Filtros")
        tanques_disponibles = sorted(df['Tanque'].unique().tolist())
        tanque_sel = st.selectbox("Tanque:", tanques_disponibles)
        st.markdown("---")

        st.subheader("⚙️ Umbral de alerta")
        umbral = st.slider("Probabilidad mínima:",
                           min_value=0.30, max_value=0.70,
                           value=0.40, step=0.05)
        st.markdown("---")
        st.caption(f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')}")

    # ── PESTAÑAS ──────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Monitoreo",
        "🔴 Tiempo Real",
        "🤖 Clasificación",
        "📈 Regresión",
        "🔮 Variables",
        "🕐 Horizontes"
    ])

    # ══════════════════════════════════════════════════════════
    # PESTAÑA 1: MONITOREO
    # ══════════════════════════════════════════════════════════
    with tab1:
        st.title("📊 Monitoreo RAS - Tilapia")
        st.markdown("**Sistema de monitoreo y predicción de eventos críticos**")
        st.markdown("---")

        df_filtrado = df[df['Tanque'] == tanque_sel].copy()
        total         = len(df_filtrado)
        criticos      = (df_filtrado['estado'] == 2).sum()
        suboptimos    = (df_filtrado['estado'] == 1).sum()
        optimos       = (df_filtrado['estado'] == 0).sum()
        ph_promedio   = df_filtrado['pH'].mean()
        temp_promedio = df_filtrado['Temperatura_C'].mean()

        # Métricas
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("📋 Registros",       total)
        col2.metric("🟢 Óptimos",         int(optimos),
                    f"{optimos/total*100:.1f}%")
        col3.metric("🟡 Subóptimos",      int(suboptimos),
                    f"{suboptimos/total*100:.1f}%", delta_color="inverse")
        col4.metric("🔴 Críticos",        int(criticos),
                    f"{criticos/total*100:.1f}%",   delta_color="inverse")
        col5.metric("💧 pH / 🌡️ Temp",
                    f"{ph_promedio:.2f} / {temp_promedio:.1f}°C")

        st.markdown("---")

        # ── Gauges de pH y Temperatura ────────────────────────
        st.subheader("🎯 Estado actual del sistema")
        ultima = df_filtrado.iloc[-1]

        col_g1, col_g2, col_g3 = st.columns(3)

        with col_g1:
            gauge_ph = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=ultima['pH'],
                title={'text': "pH Actual", 'font': {'color': '#E3F2FD'}},
                delta={'reference': 7.0, 'relative': False},
                gauge={
                    'axis': {'range': [5, 10],
                             'tickcolor': '#7FB3D3'},
                    'bar': {'color': "#1565C0"},
                    'bgcolor': "#0D2137",
                    'steps': [
                     {'range': [5.0,  6.05], 'color': '#D32F2F'},
                     {'range': [6.05, 6.5],  'color': '#FBC02D'},
                     {'range': [6.5,  8.5],  'color': '#388E3C'},
                     {'range': [8.5,  9.0],  'color': '#FBC02D'},
                     {'range': [9.0,  10.0], 'color': '#D32F2F'},
                     ],
                    'threshold': {
                        'line': {'color': "white", 'width': 3},
                        'thickness': 0.75,
                        'value': ultima['pH']
                    }
                },
                number={'font': {'color': '#E3F2FD', 'size': 24}}
            ))
            gauge_ph.update_layout(
                height=280, paper_bgcolor='#0A1628',
                font=dict(color='#E3F2FD')
            )
            st.plotly_chart(gauge_ph, use_container_width=True)

        with col_g2:
            gauge_temp = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=ultima['Temperatura_C'],
                title={'text': "Temperatura (°C)",
                       'font': {'color': '#E3F2FD'}},
                delta={'reference': 18.0, 'relative': False},
                gauge={
                    'axis': {'range': [8, 32],
                             'tickcolor': '#7FB3D3'},
                    'bar': {'color': "#C62828"},
                    'bgcolor': "#0D2137",
                    'steps': [
                     {'range': [8,   11],  'color': '#D32F2F'},
                     {'range': [11,  16],  'color': '#FBC02D'},
                     {'range': [16,  20],  'color': '#388E3C'},
                     {'range': [20,  27],  'color': '#FBC02D'},
                     {'range': [27,  32],  'color': '#D32F2F'},
                     ],
                    'threshold': {
                        'line': {'color': "white", 'width': 3},
                        'thickness': 0.75,
                        'value': ultima['Temperatura_C']
                    }
                },
                number={'suffix': '°C',
                        'font': {'color': '#E3F2FD', 'size': 24}}
            ))
            gauge_temp.update_layout(
                height=280, paper_bgcolor='#0A1628',
                font=dict(color='#E3F2FD')
            )
            st.plotly_chart(gauge_temp, use_container_width=True)

        with col_g3:
            # Distribución de estados (dona)
            fig_dona = go.Figure(go.Pie(
                labels=['🟢 Óptimo', '🟡 Subóptimo', '🔴 Crítico'],
                values=[int(optimos), int(suboptimos), int(criticos)],
                hole=0.6,
                marker=dict(colors=['#2E7D32', '#F57C00', '#C62828'],
                            line=dict(color='#0A1628', width=2))
            ))
            fig_dona.update_layout(
                title=dict(text='Distribución de estados',
                           font=dict(color='#E3F2FD')),
                height=280,
                paper_bgcolor='#0A1628',
                font=dict(color='#E3F2FD'),
                showlegend=True,
                legend=dict(font=dict(color='#E3F2FD'))
            )
            st.plotly_chart(fig_dona, use_container_width=True)

        st.markdown("---")

        # ── Indicador de tendencia ────────────────────────────
        st.subheader("📉 Tendencia reciente")
        ultimas4 = df_filtrado.tail(4)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            if len(ultimas4) >= 2:
                diff_ph = ultimas4['pH'].iloc[-1] - ultimas4['pH'].iloc[-2]
                tendencia_ph = "📈 Subiendo" if diff_ph > 0.05 else \
                               "📉 Bajando"  if diff_ph < -0.05 else \
                               "➡️ Estable"
                color_ph = "#F44336" if abs(diff_ph) > 0.2 else \
                           "#FF9800" if abs(diff_ph) > 0.05 else "#4CAF50"
                st.markdown(f"""
                    <div class='tanque-card'>
                        <h4 style='color:#7FB3D3'>💧 Tendencia pH</h4>
                        <h2 style='color:{color_ph}'>{tendencia_ph}</h2>
                        <p style='color:#E3F2FD'>
                            Cambio: {diff_ph:+.3f} unidades
                        </p>
                    </div>
                """, unsafe_allow_html=True)

        with col_t2:
            if len(ultimas4) >= 2:
                diff_temp = (ultimas4['Temperatura_C'].iloc[-1] -
                             ultimas4['Temperatura_C'].iloc[-2])
                tendencia_temp = "📈 Subiendo" if diff_temp > 0.2 else \
                                 "📉 Bajando"  if diff_temp < -0.2 else \
                                 "➡️ Estable"
                color_temp = "#F44336" if abs(diff_temp) > 1.0 else \
                             "#FF9800" if abs(diff_temp) > 0.2 else "#4CAF50"
                st.markdown(f"""
                    <div class='tanque-card'>
                        <h4 style='color:#7FB3D3'>🌡️ Tendencia Temperatura</h4>
                        <h2 style='color:{color_temp}'>{tendencia_temp}</h2>
                        <p style='color:#E3F2FD'>
                            Cambio: {diff_temp:+.2f}°C
                        </p>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Evolución histórica ───────────────────────────────
        st.subheader(f"📈 Evolución histórica - {tanque_sel}")
        df_crit = df_filtrado[df_filtrado['estado'] == 2]

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=('pH', 'Temperatura (°C)'),
            vertical_spacing=0.12
        )
        fig.add_trace(go.Scatter(
            x=df_filtrado['Fecha_Hora'], y=df_filtrado['pH'],
            mode='lines', name='pH',
            line=dict(color='#42A5F5', width=2)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_crit['Fecha_Hora'], y=df_crit['pH'],
            mode='markers', name='Crítico',
            marker=dict(color='#F44336', size=8, symbol='x')
        ), row=1, col=1)
        fig.add_hrect(y0=6.5, y1=8.5, fillcolor="#1B5E20",
                      opacity=0.1, row=1, col=1,
                      annotation_text="Zona óptima",
                      annotation_font_color="#4CAF50")
        fig.add_trace(go.Scatter(
            x=df_filtrado['Fecha_Hora'],
            y=df_filtrado['Temperatura_C'],
            mode='lines', name='Temperatura',
            line=dict(color='#EF5350', width=2)
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df_crit['Fecha_Hora'],
            y=df_crit['Temperatura_C'],
            mode='markers', name='Crítico Temp',
            marker=dict(color='#F44336', size=8, symbol='x')
        ), row=2, col=1)
        fig.add_hrect(y0=16.0, y1=20.0, fillcolor="#1B5E20",
                      opacity=0.1, row=2, col=1,
                      annotation_text="Zona óptima",
                      annotation_font_color="#4CAF50")
        fig.update_layout(
            height=500, plot_bgcolor='#0D2137',
            paper_bgcolor='#0A1628',
            font=dict(color='#E3F2FD'),
            legend=dict(font=dict(color='#E3F2FD'))
        )
        fig.update_xaxes(gridcolor='#1E3A5F')
        fig.update_yaxes(gridcolor='#1E3A5F')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── Correlación pH vs Temperatura ────────────────────
        st.subheader("🔗 Correlación pH vs Temperatura")
        colores_estado = df_filtrado['estado'].map(
            {0: '#4CAF50', 1: '#FF9800', 2: '#F44336'})

        fig_corr = go.Figure()
        for estado, color, label in [
            (0, '#4CAF50', '🟢 Óptimo'),
            (1, '#FF9800', '🟡 Subóptimo'),
            (2, '#F44336', '🔴 Crítico')
        ]:
            df_e = df_filtrado[df_filtrado['estado'] == estado]
            if len(df_e) > 0:
                fig_corr.add_trace(go.Scatter(
                    x=df_e['pH'],
                    y=df_e['Temperatura_C'],
                    mode='markers',
                    name=label,
                    marker=dict(color=color, size=7, opacity=0.7)
                ))

        fig_corr.add_hrect(y0=16, y1=20, fillcolor="#1B5E20",
                           opacity=0.08)
        fig_corr.add_vrect(x0=6.5, x1=8.5, fillcolor="#1B5E20",
                           opacity=0.08,
                           annotation_text="Zona óptima",
                           annotation_font_color="#4CAF50")
        fig_corr.update_layout(
            xaxis_title='pH',
            yaxis_title='Temperatura (°C)',
            plot_bgcolor='#0D2137',
            paper_bgcolor='#0A1628',
            font=dict(color='#E3F2FD'),
            height=400
        )
        fig_corr.update_xaxes(gridcolor='#1E3A5F')
        fig_corr.update_yaxes(gridcolor='#1E3A5F')
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("---")

        # ── Heatmap de todos los tanques ──────────────────────
        st.subheader("🗺️ Heatmap de estado por tanque")
        df_heat = df.groupby('Tanque').agg(
            pH_prom=('pH', 'mean'),
            Temp_prom=('Temperatura_C', 'mean'),
            Criticos=('estado', lambda x: (x == 2).sum()),
            Total=('estado', 'count')
        ).reset_index()
        df_heat['Pct_Criticos'] = (
            df_heat['Criticos'] / df_heat['Total'] * 100
        )

        fig_heat = go.Figure(go.Bar(
            x=df_heat['Tanque'],
            y=df_heat['Pct_Criticos'],
            marker=dict(
                color=df_heat['Pct_Criticos'],
                colorscale=[[0, '#1B5E20'],
                            [0.5, '#F57C00'],
                            [1, '#B71C1C']],
                showscale=True,
                colorbar=dict(
                    title='% Críticos',
                    tickfont=dict(color='#E3F2FD'),
                    title_font=dict(color='#E3F2FD')
                )
            ),
            text=df_heat['Pct_Criticos'].apply(lambda x: f"{x:.1f}%"),
            textposition='outside',
            textfont=dict(color='#E3F2FD')
        ))
        fig_heat.update_layout(
            xaxis_title='Tanque',
            yaxis_title='% Eventos Críticos',
            plot_bgcolor='#0D2137',
            paper_bgcolor='#0A1628',
            font=dict(color='#E3F2FD'),
            height=350
        )
        fig_heat.update_xaxes(gridcolor='#1E3A5F',
                               tickangle=45)
        fig_heat.update_yaxes(gridcolor='#1E3A5F')
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")

        # ── Predicción manual ─────────────────────────────────
        st.subheader("🔮 Predicción en tiempo real")
        col_izq, col_der = st.columns([1, 1])

        with col_izq:
            tanque_pred  = st.selectbox("Tanque:", tanques_disponibles,
                                         key='pred_tanque')
            ph_input     = st.number_input("pH:", min_value=0.0,
                                            max_value=14.0,
                                            value=7.2, step=0.1)
            temp_input   = st.number_input("Temperatura (°C):",
                                            min_value=0.0,
                                            max_value=50.0,
                                            value=18.0, step=0.1)
            predecir_btn = st.button("🔍 Predecir",
                                      use_container_width=True)

        with col_der:
            if predecir_btn:
                hist = df[df['Tanque'] == tanque_pred].tail(4)
                if len(hist) > 0:
                    ph_ant    = hist['pH'].iloc[-1]
                    temp_ant  = hist['Temperatura_C'].iloc[-1]
                    ph_diff   = ph_input - ph_ant
                    temp_diff = temp_input - temp_ant
                    ph_prom   = (hist['pH'].mean() + ph_input) / 2
                    temp_prom = (hist['Temperatura_C'].mean()
                                 + temp_input) / 2
                else:
                    ph_diff = temp_diff = 0
                    ph_prom = ph_input
                    temp_prom = temp_input

                hora_actual = datetime.now().hour
                jornada_num = 0 if hora_actual < 12 else 1
                tanque_num  = int(tanque_pred.split()[-1])

                datos_pred = pd.DataFrame([[
                    ph_input, temp_input, jornada_num, tanque_num,
                    ph_diff, temp_diff, ph_prom, temp_prom
                ]], columns=[
                    'pH', 'Temperatura_C', 'Jornada_num', 'Tanque_num',
                    'pH_diff', 'temp_diff', 'pH_promedio', 'temp_promedio'
                ])

                datos_scaled   = scaler.transform(datos_pred)
                probabilidades = modelo.predict_proba(datos_scaled)[0]
                prob_suboptimo = probabilidades[1]
                prob_critico   = probabilidades[2]
                probabilidad   = max(prob_suboptimo, prob_critico)
                estado_pred    = modelo.predict(datos_scaled)[0]

                alerta_directa = (
                    ph_input < 6.05 or ph_input > 9.0 or
                    temp_input < 11.0 or temp_input > 27.0
                )
                if alerta_directa:
                    probabilidad = max(probabilidad, 0.95)

                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probabilidad * 100,
                    title={'text': "Probabilidad evento crítico",
                           'font': {'color': '#E3F2FD'}},
                    gauge={
                        'axis': {'range': [0, 100],
                                 'tickcolor': '#7FB3D3'},
                        'bar':  {'color': "#1565C0"},
                        'bgcolor': "#0D2137",
                        'steps': [
                            {'range': [0,  40], 'color': '#1B5E20'},
                            {'range': [40, 70], 'color': '#E65100'},
                            {'range': [70,100], 'color': '#B71C1C'}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 3},
                            'thickness': 0.75,
                            'value': umbral * 100
                        }
                    },
                    number={'suffix': '%',
                            'font': {'color': '#E3F2FD', 'size': 40}}
                ))
                gauge.update_layout(
                    height=280, paper_bgcolor='#0A1628',
                    font=dict(color='#E3F2FD')
                )
                st.plotly_chart(gauge, use_container_width=True)

                if estado_pred == 2 or alerta_directa:
                    st.markdown(
                        '<div class="alerta-roja">'
                        '🔴 CRÍTICO — ¡Intervenir de inmediato!</div>',
                        unsafe_allow_html=True)
                elif estado_pred == 1:
                    st.markdown(
                        '<div class="alerta-amarilla">'
                        '🟡 SUBÓPTIMO — Revisar pronto</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="alerta-verde">'
                        '🟢 ÓPTIMO — Todo en orden</div>',
                        unsafe_allow_html=True)

                st.markdown("---")
                col_p1, col_p2, col_p3 = st.columns(3)
                col_p1.metric("🟢 Óptimo",
                              f"{probabilidades[0]*100:.1f}%")
                col_p2.metric("🟡 Subóptimo",
                              f"{probabilidades[1]*100:.1f}%")
                col_p3.metric("🔴 Crítico",
                              f"{probabilidades[2]*100:.1f}%")

    # ══════════════════════════════════════════════════════════
    # PESTAÑA 2: TIEMPO REAL
    # ══════════════════════════════════════════════════════════
    with tab2:
        st.title("🔴 Monitoreo en Tiempo Real")
        st.markdown("**Lecturas simuladas actualizándose automáticamente**")
        st.markdown("---")

        with st.expander("⚙️ ¿Cómo activar el simulador?"):
            st.code("python simulador_tiempo_real.py", language="bash")
            st.markdown("Ejecuta ese comando en una terminal aparte "
                        "y mantén ambas abiertas.")

        intervalo = st.slider("Intervalo de actualización (segundos):",
                               5, 30, 10)

        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=intervalo * 1000, key="refresh_tr")
        except:
            st.info("💡 Para auto-actualización: "
                    "`pip install streamlit-autorefresh`")
            if st.button("🔄 Actualizar manualmente"):
                st.rerun()

        ARCHIVO_TR = 'datos/lecturas_tiempo_real.csv'

        if not os.path.exists(ARCHIVO_TR):
            st.warning("⚠️ Ejecuta `python simulador_tiempo_real.py` primero.")
        else:
            df_tr = pd.read_csv(ARCHIVO_TR)
            df_tr['Timestamp'] = pd.to_datetime(df_tr['Timestamp'])

            if len(df_tr) == 0:
                st.warning("⚠️ Archivo vacío.")
            else:
                df_ultima = (df_tr.sort_values('Timestamp')
                             .groupby('Tanque').last().reset_index())

                total_crit  = (df_ultima['Estado'] == 2).sum()
                total_subop = (df_ultima['Estado'] == 1).sum()
                total_opt   = (df_ultima['Estado'] == 0).sum()
                ultima_hora = df_tr['Timestamp'].max().strftime('%H:%M:%S')

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🕐 Última lectura",  ultima_hora)
                col2.metric("🟢 Óptimos",         int(total_opt))
                col3.metric("🟡 Subóptimos",      int(total_subop))
                col4.metric("🔴 Críticos",        int(total_crit),
                            delta_color="inverse")

                st.markdown("---")
                st.subheader("🐟 Estado actual por tanque")
                cols_tanques = st.columns(5)
                etiquetas = {
                    0: ("🟢", "ÓPTIMO",    "alerta-verde"),
                    1: ("🟡", "SUBÓPTIMO", "alerta-amarilla"),
                    2: ("🔴", "CRÍTICO",   "alerta-roja")
                }

                for i, row in df_ultima.iterrows():
                    col_idx = i % 5
                    icono, texto, clase = etiquetas[row['Estado']]
                    with cols_tanques[col_idx]:
                        st.markdown(
                            f'<div class="{clase}" '
                            f'style="margin-bottom:10px">'
                            f'<b>{row["Tanque"]}</b><br>'
                            f'{icono} {texto}<br>'
                            f'pH: {row["pH"]:.2f}<br>'
                            f'T: {row["Temperatura_C"]:.1f}°C'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                st.markdown("---")
                st.subheader("📈 Evolución en tiempo real")
                tanque_tr = st.selectbox(
                    "Tanque:", sorted(df_tr['Tanque'].unique()),
                    key='tr_tanque'
                )
                df_tr_fil = df_tr[df_tr['Tanque'] == tanque_tr].tail(50)

                fig_tr = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=('pH', 'Temperatura (°C)'),
                    vertical_spacing=0.12
                )
                colores_e = df_tr_fil['Estado'].map(
                    {0: '#4CAF50', 1: '#FF9800', 2: '#F44336'})

                fig_tr.add_trace(go.Scatter(
                    x=df_tr_fil['Timestamp'], y=df_tr_fil['pH'],
                    mode='lines+markers', name='pH',
                    line=dict(color='#42A5F5', width=2),
                    marker=dict(color=colores_e, size=8)
                ), row=1, col=1)
                fig_tr.add_trace(go.Scatter(
                    x=df_tr_fil['Timestamp'],
                    y=df_tr_fil['Temperatura_C'],
                    mode='lines+markers', name='Temperatura',
                    line=dict(color='#EF5350', width=2),
                    marker=dict(color=colores_e, size=8)
                ), row=2, col=1)
                fig_tr.update_layout(
                    height=450, plot_bgcolor='#0D2137',
                    paper_bgcolor='#0A1628',
                    font=dict(color='#E3F2FD')
                )
                fig_tr.update_xaxes(gridcolor='#1E3A5F')
                fig_tr.update_yaxes(gridcolor='#1E3A5F')
                st.plotly_chart(fig_tr, use_container_width=True)

                st.markdown("---")
                st.subheader("📋 Últimas lecturas")
                df_tabla = (df_tr.sort_values('Timestamp', ascending=False)
                            .head(20).copy())
                df_tabla['Estado'] = df_tabla['Estado'].map(
                    {0: '🟢 Óptimo', 1: '🟡 Subóptimo', 2: '🔴 Crítico'})
                st.dataframe(df_tabla[['Timestamp', 'Tanque', 'pH',
                                        'Temperatura_C', 'Estado']],
                             use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════
    # PESTAÑA 3: CLASIFICACIÓN
    # ══════════════════════════════════════════════════════════
    with tab3:
        st.title("🤖 Comparación de Modelos de Clasificación")
        st.markdown("---")

        df_clf = pd.DataFrame(resultados['clasificacion'])

        st.subheader("📋 Tabla comparativa")
        df_mostrar = df_clf.copy()
        df_mostrar['Accuracy']   = df_mostrar['Accuracy'].apply(
                                   lambda x: f"{x*100:.1f}%")
        df_mostrar['F1 Macro']   = df_mostrar['F1 Macro'].apply(
                                   lambda x: f"{x*100:.1f}%")
        df_mostrar['F1 Crítico'] = df_mostrar['F1 Crítico'].apply(
                                   lambda x: f"{x*100:.1f}%")
        df_mostrar['CV Media']   = df_mostrar['CV Media'].apply(
                                   lambda x: f"{x*100:.1f}%" if x > 0
                                   else "N/A")
        st.dataframe(df_mostrar[['Modelo', 'Accuracy', 'F1 Macro',
                                  'F1 Crítico', 'CV Media']],
                     use_container_width=True, hide_index=True)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Accuracy por modelo")
            fig_acc = px.bar(
                df_clf, x='Modelo', y='Accuracy', color='Modelo',
                color_discrete_sequence=['#1565C0','#1976D2',
                                         '#1E88E5','#42A5F5'],
                text=df_clf['Accuracy'].apply(lambda x: f"{x*100:.1f}%")
            )
            fig_acc.update_traces(textposition='outside')
            fig_acc.update_layout(
                plot_bgcolor='#0D2137', paper_bgcolor='#0A1628',
                font=dict(color='#E3F2FD'), showlegend=False,
                yaxis=dict(tickformat='.0%'), height=350
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with col2:
            st.subheader("📊 F1-Score Macro")
            fig_f1 = px.bar(
                df_clf, x='Modelo', y='F1 Macro', color='Modelo',
                color_discrete_sequence=['#1565C0','#1976D2',
                                         '#1E88E5','#42A5F5'],
                text=df_clf['F1 Macro'].apply(lambda x: f"{x*100:.1f}%")
            )
            fig_f1.update_traces(textposition='outside')
            fig_f1.update_layout(
                plot_bgcolor='#0D2137', paper_bgcolor='#0A1628',
                font=dict(color='#E3F2FD'), showlegend=False,
                yaxis=dict(tickformat='.0%'), height=350
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        st.markdown("---")
        st.subheader("🔢 Matrices de confusión")
        matrices = resultados['matrices_conf']
        cols     = st.columns(len(matrices))

        for i, (nombre, matriz) in enumerate(matrices.items()):
            with cols[i]:
                fig_cm = px.imshow(
                    matriz,
                    labels=dict(x="Predicho", y="Real", color="Count"),
                    x=['Óptimo', 'Subóptimo', 'Crítico'],
                    y=['Óptimo', 'Subóptimo', 'Crítico'],
                    color_continuous_scale='Blues',
                    text_auto=True, title=nombre
                )
                fig_cm.update_layout(
                    plot_bgcolor='#0D2137', paper_bgcolor='#0A1628',
                    font=dict(color='#E3F2FD'), height=300
                )
                st.plotly_chart(fig_cm, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # PESTAÑA 4: REGRESIÓN
    # ══════════════════════════════════════════════════════════
    with tab4:
        st.title("📈 Comparación de Modelos de Regresión")
        st.markdown("---")

        df_reg = pd.DataFrame(resultados['regresion'])

        st.subheader("📋 Tabla comparativa")
        df_reg_mostrar = df_reg.copy()
        for col in ['MAE pH', 'RMSE pH', 'MAE Temp', 'RMSE Temp']:
            df_reg_mostrar[col] = df_reg_mostrar[col].apply(
                                  lambda x: f"{x:.4f}")
        for col in ['R² pH', 'R² Temp']:
            df_reg_mostrar[col] = df_reg_mostrar[col].apply(
                                  lambda x: f"{x:.3f}")
        st.dataframe(df_reg_mostrar, use_container_width=True,
                     hide_index=True)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("R² - pH")
            fig_r2ph = px.bar(
                df_reg, x='Modelo', y='R² pH', color='Modelo',
                color_discrete_sequence=['#1565C0','#1976D2','#1E88E5',
                                         '#42A5F5'],
                text=df_reg['R² pH'].apply(lambda x: f"{x:.3f}")
            )
            fig_r2ph.update_traces(textposition='outside')
            fig_r2ph.update_layout(
                plot_bgcolor='#0D2137', paper_bgcolor='#0A1628',
                font=dict(color='#E3F2FD'), showlegend=False, height=350
            )
            st.plotly_chart(fig_r2ph, use_container_width=True)

        with col2:
            st.subheader("R² - Temperatura")
            fig_r2t = px.bar(
                df_reg, x='Modelo', y='R² Temp', color='Modelo',
                color_discrete_sequence=['#1565C0','#1976D2','#1E88E5',
                                         '#42A5F5'],
                text=df_reg['R² Temp'].apply(lambda x: f"{x:.3f}")
            )
            fig_r2t.update_traces(textposition='outside')
            fig_r2t.update_layout(
                plot_bgcolor='#0D2137', paper_bgcolor='#0A1628',
                font=dict(color='#E3F2FD'), showlegend=False, height=350
            )
            st.plotly_chart(fig_r2t, use_container_width=True)

        st.markdown("---")
        st.subheader("🎯 Real vs Predicho")
        modelo_sel = st.selectbox(
            "Seleccionar modelo:",
            [r['Modelo'] for r in resultados['regresion']
             if r['Modelo'] in resultados['pred_reg']]
        )

        if modelo_sel in resultados['pred_reg']:
            pred = resultados['pred_reg'][modelo_sel]
            col1, col2 = st.columns(2)

            with col1:
                fig_rvp_ph = go.Figure()
                fig_rvp_ph.add_trace(go.Scatter(
                    x=pred['ph_real'], y=pred['ph_pred'],
                    mode='markers',
                    marker=dict(color='#42A5F5', size=6, opacity=0.6),
                    name='Predicciones'
                ))
                min_ph = min(pred['ph_real'])
                max_ph = max(pred['ph_real'])
                fig_rvp_ph.add_trace(go.Scatter(
                    x=[min_ph, max_ph], y=[min_ph, max_ph],
                    mode='lines',
                    line=dict(color='#F44336', dash='dash', width=2),
                    name='Predicción perfecta'
                ))
                fig_rvp_ph.update_layout(
                    title='Real vs Predicho - pH',
                    xaxis_title='pH Real', yaxis_title='pH Predicho',
                    plot_bgcolor='#0D2137', paper_bgcolor='#0A1628',
                    font=dict(color='#E3F2FD'), height=350
                )
                st.plotly_chart(fig_rvp_ph, use_container_width=True)

            with col2:
                fig_rvp_t = go.Figure()
                fig_rvp_t.add_trace(go.Scatter(
                    x=pred['temp_real'], y=pred['temp_pred'],
                    mode='markers',
                    marker=dict(color='#EF5350', size=6, opacity=0.6),
                    name='Predicciones'
                ))
                min_t = min(pred['temp_real'])
                max_t = max(pred['temp_real'])
                fig_rvp_t.add_trace(go.Scatter(
                    x=[min_t, max_t], y=[min_t, max_t],
                    mode='lines',
                    line=dict(color='#F44336', dash='dash', width=2),
                    name='Predicción perfecta'
                ))
                fig_rvp_t.update_layout(
                    title='Real vs Predicho - Temperatura',
                    xaxis_title='Temperatura Real (°C)',
                    yaxis_title='Temperatura Predicha (°C)',
                    plot_bgcolor='#0D2137', paper_bgcolor='#0A1628',
                    font=dict(color='#E3F2FD'), height=350
                )
                st.plotly_chart(fig_rvp_t, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # PESTAÑA 5: IMPORTANCIA DE VARIABLES
    # ══════════════════════════════════════════════════════════
    with tab5:
        st.title("🔮 Importancia de Variables")
        st.markdown("**Basado en Random Forest**")
        st.markdown("---")

        importancias = resultados['importancias']
        df_imp = pd.DataFrame({
            'Variable':    list(importancias.keys()),
            'Importancia': list(importancias.values())
        }).sort_values('Importancia', ascending=True)

        fig_imp = px.bar(
            df_imp, x='Importancia', y='Variable',
            orientation='h', color='Importancia',
            color_continuous_scale=[[0, '#1565C0'], [1, '#42A5F5']],
            text=df_imp['Importancia'].apply(lambda x: f"{x*100:.1f}%")
        )
        fig_imp.update_traces(textposition='outside')
        fig_imp.update_layout(
            plot_bgcolor='#0D2137', paper_bgcolor='#0A1628',
            font=dict(color='#E3F2FD'), height=450, showlegend=False
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("---")
        st.subheader("💡 Interpretación")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Variables más importantes:**
            - **pH y Temperatura actuales** → el estado presente
              es el mejor predictor del futuro
            - **pH_promedio y temp_promedio** → la tendencia
              reciente es muy informativa
            """)
        with col2:
            st.markdown("""
            **Variables menos importantes:**
            - **Jornada** → mañana o tarde no cambia mucho
              el comportamiento del sistema
            - **Tanque_num** → todos los tanques se comportan
              de forma similar
            """)

    # ══════════════════════════════════════════════════════════
    # PESTAÑA 6: HORIZONTES
    # ══════════════════════════════════════════════════════════
    with tab6:
        st.title("🕐 Horizontes de Predicción")
        st.markdown("**¿Con cuánta anticipación puede predecir el modelo?**")
        st.markdown("---")

        df_hor = pd.DataFrame(resultados['horizontes'])

        st.subheader("📋 Resultados por horizonte")
        df_hor_mostrar = df_hor.copy()
        for col in ['Accuracy', 'F1 Macro', 'F1 Crítico']:
            df_hor_mostrar[col] = df_hor_mostrar[col].apply(
                                  lambda x: f"{x*100:.1f}%")
        st.dataframe(df_hor_mostrar[['Horizonte', 'Accuracy',
                                      'F1 Macro', 'F1 Crítico']],
                     use_container_width=True, hide_index=True)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            fig_hor_acc = go.Figure()
            fig_hor_acc.add_trace(go.Scatter(
                x=df_hor['Horas'], y=df_hor['Accuracy'],
                mode='lines+markers+text',
                text=df_hor['Accuracy'].apply(lambda x: f"{x*100:.1f}%"),
                textposition='top center',
                line=dict(color='#42A5F5', width=3),
                marker=dict(size=10, color='#42A5F5')
            ))
            fig_hor_acc.update_layout(
                title='Accuracy vs Horizonte',
                xaxis_title='Horizonte (horas)',
                yaxis_title='Accuracy',
                xaxis=dict(tickvals=df_hor['Horas'].tolist()),
                plot_bgcolor='#0D2137', paper_bgcolor='#0A1628',
                font=dict(color='#E3F2FD'), height=350,
                yaxis=dict(tickformat='.0%')
            )
            st.plotly_chart(fig_hor_acc, use_container_width=True)

        with col2:
            fig_hor_f1 = go.Figure()
            fig_hor_f1.add_trace(go.Scatter(
                x=df_hor['Horas'], y=df_hor['F1 Crítico'],
                mode='lines+markers+text',
                text=df_hor['F1 Crítico'].apply(
                    lambda x: f"{x*100:.1f}%"),
                textposition='top center',
                line=dict(color='#EF5350', width=3),
                marker=dict(size=10, color='#EF5350'),
                name='F1 Crítico'
            ))
            fig_hor_f1.update_layout(
                title='F1 Crítico vs Horizonte',
                xaxis_title='Horizonte (horas)',
                yaxis_title='F1 Crítico',
                xaxis=dict(tickvals=df_hor['Horas'].tolist()),
                plot_bgcolor='#0D2137', paper_bgcolor='#0A1628',
                font=dict(color='#E3F2FD'), height=350,
                yaxis=dict(tickformat='.0%')
            )
            st.plotly_chart(fig_hor_f1, use_container_width=True)

        st.markdown("---")
        st.subheader("💡 Interpretación")
        st.markdown("""
        A medida que aumenta el horizonte la precisión **baja gradualmente**.
        Esto es completamente normal porque predecir más lejos en el futuro
        siempre es más difícil.

        **Recomendación:** El horizonte óptimo para tu sistema será entre
        **3 y 6 horas**, dando tiempo suficiente para intervenir sin
        perder demasiada precisión.
        """)
#Los usuarios y contraseñas son:

#Usuario: admin          Contraseña: admin123
#Usuario: investigador   Contraseña: ras2024

#Usuario: estudiante     Contraseña: udca2024




