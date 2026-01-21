#!/usr/bin/env python3
"""
Kirigami Air-to-Water Converter Dashboard
Based on the calculation framework by Hitendra Vaishnav (January 19, 2026)
"""

import streamlit as st
import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw
import cv2
import io
import base64

# Set page config
st.set_page_config(
    page_title="Kirigami Air-to-Water Dashboard", 
    layout="wide"
)

# Custom CSS for blue theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #e6f0ff 0%, #d1e0ff 100%);
    }
    
    .main-header {
        color: #1a237e;
        text-align: center;
        font-weight: 700;
        margin-bottom: 1rem;
        font-size: 2.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        color: #283593;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #1a56db;
        padding-bottom: 5px;
    }
    
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #1a56db;
        height: 100%;
    }
    
    .stMetric {
        background-color: transparent;
    }
    
    .stSidebar {
        background-color: #f0f7ff;
        box-shadow: 2px 0 5px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background-color: #1a56db;
        color: white;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #1546b8;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .success-box {
        background-color: #e0f2fe;
        border-left: 4px solid #0369a1;
        padding: 1.2rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #d97706;
        padding: 1.2rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: #dbeafe;
        border-left: 4px solid #1d4ed8;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    
    .section-box {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    
    /* Custom slider styling */
    .stSlider > div > div > div {
        background-color: #1a56db !important;
    }
    
    .stSlider > div > div > div > div {
        background-color: #1a56db !important;
    }
    
    /* Sidebar widgets */
    .sidebar .widget-content {
        background-color: #f8fafc;
    }
    
    /* Input fields */
    .stNumberInput > div > div > input {
        border-color: #1a56db;
    }
    
    .stSelectbox > div > div > div {
        border-color: #1a56db;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-color: #1a56db !important;
    }
    
    .dataframe th {
        background-color: #1e40af !important;
        color: white !important;
    }
    
    .dataframe tr:nth-child(even) {
        background-color: #eff6ff !important;
    }
</style>
""", unsafe_allow_html=True)

P_ATM = 101325
MU_AIR = 1.8e-5
K_AIR = 0.026
CP_AIR = 1006
PR_AIR = 0.71
LE = 1.0
R_DA = 287.05
HFG = 2.45e6
RHO_WATER = 1000
K_AL = 205

@dataclass
class KirigamiGeometry:
    L_flow: float
    W: float
    H_face: float
    theta: float
    phi: float
    d_h: float
    N_rows: int

@dataclass
class FinGeometry:
    A_base: float
    A_fin: float
    t_fin: float
    L_f: float

@dataclass
class PeltierParams:
    P_elec: float = 15.0
    COP_eff: float = 0.4
    R_th_h: float = 2.0
    delta_T_max: float = 50.0
    Q_c_max: float = 10.0

def psat_magnus_pa(T_c: float) -> float:
    return 610.94 * math.exp((17.625 * T_c) / (T_c + 243.04))

def humidity_ratio(P_v: float, P: float = P_ATM) -> float:
    return 0.62198 * P_v / (P - P_v)

def moist_air_density(T_c: float, w: float, P: float = P_ATM) -> float:
    T_K = T_c + 273.15
    return P / (R_DA * T_K) * (1 / (1 + 1.6078 * w))

def friction_factor(Re: float) -> float:
    if Re < 2300:
        return 64 / Re
    elif Re > 4000:
        return 0.3164 * Re**(-0.25)
    else:
        f_lam = 64 / 2300
        f_turb = 0.3164 * 4000**(-0.25)
        return f_lam + (f_turb - f_lam) * (Re - 2300) / (4000 - 2300)

def kirigami_loss_coefficient(phi: float, theta: float, N_rows: int, 
                              C1: float = 1.0, C2: float = 2.0) -> float:
    K_in = 0.5
    K_out = 1.0
    K_kir = C1 * ((1 - phi) / phi**2) * (1 + C2 * math.sin(math.radians(theta))**2) * N_rows
    return K_in + K_out + K_kir

def pressure_drop(rho: float, v: float, geom: KirigamiGeometry, 
                 C1: float = 1.0, C2: float = 2.0) -> Tuple[float, float]:
    Re = rho * v * geom.d_h / MU_AIR
    f = friction_factor(Re)
    K_total = kirigami_loss_coefficient(geom.phi, geom.theta, geom.N_rows, C1, C2)
    dP = (f * geom.L_flow / geom.d_h + K_total) * rho * v**2 / 2
    return dP, Re

def nusselt_number(Re: float) -> float:
    if Re < 2300:
        return 3.66
    else:
        return 0.023 * Re**0.8 * PR_AIR**0.4

def fin_efficiency(h: float, fin_geom: FinGeometry) -> Tuple[float, float]:
    m = math.sqrt(2 * h / (K_AL * fin_geom.t_fin))
    eta_f = math.tanh(m * fin_geom.L_f) / (m * fin_geom.L_f) if m * fin_geom.L_f > 0 else 1.0
    A_eff = fin_geom.A_base + eta_f * fin_geom.A_fin
    return eta_f, A_eff

def calculate_load(T_inf: float, T_s: float, w_inf: float, 
                  h: float, A_eff: float, rho: float) -> Tuple[float, float]:
    P_sat_s = psat_magnus_pa(T_s)
    w_s = humidity_ratio(P_sat_s)
    
    h_m = h / (rho * CP_AIR)
    
    if w_inf > w_s:
        m_dot_cond = rho * h_m * A_eff * (w_inf - w_s)
    else:
        m_dot_cond = 0.0
    
    Q_sensible = h * A_eff * (T_inf - T_s)
    Q_latent = m_dot_cond * HFG
    Q_load = Q_sensible + Q_latent
    
    return Q_load, m_dot_cond

def solve_surface_temperature(T_inf: float, w_inf: float, Q_c: float,
                             h: float, A_eff: float, rho: float,
                             T_min: float = -10.0, T_max: float = None) -> Tuple[float, float, float]:
    if T_max is None:
        T_max = T_inf
    
    tolerance = 0.01
    max_iter = 50
    
    for _ in range(max_iter):
        T_s = (T_min + T_max) / 2
        Q_load, m_dot_cond = calculate_load(T_inf, T_s, w_inf, h, A_eff, rho)
        
        if abs(Q_load - Q_c) < tolerance:
            break
        
        if Q_load > Q_c:
            T_min = T_s
        else:
            T_max = T_s
    
    Q_load, m_dot_cond = calculate_load(T_inf, T_s, w_inf, h, A_eff, rho)
    return T_s, m_dot_cond, Q_load

def calculate_dew_point(T_c: float, RH: float) -> float:
    P_sat = psat_magnus_pa(T_c)
    P_v = RH * P_sat
    ln_ratio = math.log(P_v / 610.94)
    T_dp = 243.04 * ln_ratio / (17.625 - ln_ratio)
    return T_dp

def detect_base_area_from_image(image: Image.Image, real_width_mm: float, real_height_mm: float) -> Tuple[float, Optional[Image.Image]]:
    """
    Detect base area from uploaded image using edge detection.
    Returns area in m² (and annotated image only if needed for display).
    """
    # Convert PIL Image to OpenCV format
    open_cv_image = np.array(image.convert('RGB'))
    img = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (assumed to be the base plate)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w_pixels, h_pixels = cv2.boundingRect(largest_contour)
        
        # Calculate pixel to mm conversion factors
        mm_per_pixel_width = real_width_mm / w_pixels if w_pixels > 0 else 0
        mm_per_pixel_height = real_height_mm / h_pixels if h_pixels > 0 else 0
        mm_per_pixel = (mm_per_pixel_width + mm_per_pixel_height) / 2
        
        # Calculate area in mm²
        area_mm2 = (w_pixels * mm_per_pixel) * (h_pixels * mm_per_pixel)
        
        # Convert to m²
        area_m2 = area_mm2 * 1e-6
        
        # Don't create annotated image unless specifically requested for display
        return area_m2, None
    
    # If no contours found, return default area
    default_area_m2 = (real_width_mm / 1000) * (real_height_mm / 1000)
    return default_area_m2, None

def run_dashboard_calculation(T_inf: float, RH: float, Q_flow: float,
                             geom: KirigamiGeometry, fin_geom: FinGeometry,
                             peltier: PeltierParams,
                             C1: float = 1.0, C2: float = 2.0) -> dict:
    
    P_sat_inf = psat_magnus_pa(T_inf)
    P_v = RH * P_sat_inf
    w_inf = humidity_ratio(P_v)
    rho = moist_air_density(T_inf, w_inf)
    m_dot_air = rho * Q_flow
    T_dp = calculate_dew_point(T_inf, RH)
    
    A_face = geom.W * geom.H_face
    A_open = geom.phi * A_face
    v = Q_flow / A_open
    dP, Re = pressure_drop(rho, v, geom, C1, C2)
    
    Nu = nusselt_number(Re)
    h = Nu * K_AIR / geom.d_h
    eta_f, A_eff = fin_efficiency(h, fin_geom)
    
    Q_c = peltier.COP_eff * peltier.P_elec
    Q_h = Q_c + peltier.P_elec
    T_h = T_inf + Q_h * peltier.R_th_h
    
    delta_T_TEC = peltier.delta_T_max * (1 - Q_c / peltier.Q_c_max)
    T_c = T_h - delta_T_TEC
    
    T_s, m_dot_cond, Q_load = solve_surface_temperature(
        T_inf, w_inf, Q_c, h, A_eff, rho
    )
    
    V_dot_water_ml_hr = m_dot_cond / RHO_WATER * 3600 * 1e6
    
    feasible = Q_c >= Q_load
    margin = Q_c - Q_load
    margin_percent = (margin / Q_c * 100) if Q_c > 0 else 0
    
    # Calculate specific humidity for display
    specific_humidity = w_inf / (1 + w_inf)
    
    return {
        'pressure_drop_pa': dP,
        'reynolds_number': Re,
        'surface_temp_c': T_s,
        'water_production_ml_hr': V_dot_water_ml_hr,
        'feasible': feasible,
        'cooling_margin_w': margin,
        'cooling_margin_percent': margin_percent,
        'cooling_capacity_w': Q_c,
        'total_load_w': Q_load,
        'heat_transfer_coeff': h,
        'effective_area_m2': A_eff,
        'fin_efficiency': eta_f,
        'air_velocity_m_s': v,
        'condensation_rate_kg_s': m_dot_cond,
        'ambient_humidity_ratio': w_inf,
        'ambient_humidity_ratio_g_per_kg': w_inf * 1000,
        'specific_humidity_g_per_kg': specific_humidity * 1000,
        'air_mass_flow_kg_s': m_dot_air,
        'dew_point_c': T_dp,
        'nusselt_number': Nu,
        'sensible_load_w': h * A_eff * (T_inf - T_s),
        'latent_load_w': m_dot_cond * HFG if m_dot_cond > 0 else 0
    }

# Main UI
st.markdown('<h1 class="main-header">Kirigami Air-to-Water Converter Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.markdown('<div class="sub-header">Input Parameters</div>', unsafe_allow_html=True)

st.sidebar.markdown("**Environmental Conditions**")
T_inf = st.sidebar.slider("Ambient Temperature (°C)", 15.0, 40.0, 25.0, 0.5)
RH = st.sidebar.slider("Relative Humidity (%)", 20.0, 100.0, 60.0, 1.0) / 100
Q_flow = st.sidebar.number_input("Volumetric Flow Rate (m³/s)", 0.001, 0.1, 0.01, 0.001, format="%.4f")

st.sidebar.markdown('<div class="sub-header">Kirigami Geometry</div>', unsafe_allow_html=True)
col1, col2 = st.sidebar.columns(2)
L_flow = col1.number_input("Flow Length (m)", 0.01, 1.0, 0.15, 0.01)
W = col2.number_input("Width (m)", 0.01, 1.0, 0.10, 0.01)
H_face = col1.number_input("Face Height (m)", 0.01, 0.5, 0.10, 0.01)
theta = col2.number_input("Flap Angle (°)", 0.0, 45.0, 23.0, 1.0)
phi = col1.slider("Open-Area Ratio", 0.3, 0.9, 0.65, 0.05)
d_h = col2.number_input("Hydraulic Dia. (mm)", 1.0, 20.0, 5.0, 0.5) / 1000

N_rows = 5

st.sidebar.markdown('<div class="sub-header">Fin Geometry Configuration</div>', unsafe_allow_html=True)

# Base area input method selection
st.sidebar.markdown("**Base Area Input Method**")
base_input_method = st.sidebar.radio(
    "Choose input method:",
    ["Manual Input", "Image Analysis"],
    label_visibility="collapsed"
)

A_base = 0.01  # Default 100 cm²

if base_input_method == "Image Analysis":
    st.sidebar.markdown("**Upload Base Plate Image**")
    uploaded_image = st.sidebar.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        # Input real dimensions
        col1, col2 = st.sidebar.columns(2)
        real_width_mm = col1.number_input("Actual Width (mm)", 10.0, 500.0, 100.0, 1.0)
        real_height_mm = col2.number_input("Actual Height (mm)", 10.0, 500.0, 100.0, 1.0)
        
        # Process image internally without displaying
        with st.spinner("Analyzing image..."):
            A_base, _ = detect_base_area_from_image(image, real_width_mm, real_height_mm)
            st.sidebar.success(f"Detected Base Area: {A_base*10000:.1f} cm²")
    else:
        # Fallback to manual input if no image uploaded
        st.sidebar.warning("No image uploaded. Using manual input.")
        A_base = st.sidebar.number_input("Base Area (cm²)", 10.0, 500.0, 100.0, 10.0) / 10000
else:
    # Manual input
    A_base = st.sidebar.number_input("Base Area (cm²)", 10.0, 500.0, 100.0, 10.0) / 10000

# Remaining fin parameters
st.sidebar.markdown("**Fin Parameters**")
A_fin = st.sidebar.number_input("Fin Area (cm²)", 50.0, 2000.0, 400.0, 50.0) / 10000
t_fin = st.sidebar.number_input("Fin Thickness (mm)", 0.5, 5.0, 1.0, 0.1) / 1000
L_f = st.sidebar.number_input("Fin Length (mm)", 5.0, 50.0, 20.0, 1.0) / 1000

st.sidebar.markdown('<div class="sub-header">Peltier Parameters</div>', unsafe_allow_html=True)
P_elec = st.sidebar.number_input("Electrical Power (W)", 5.0, 50.0, 15.0, 1.0)
COP_eff = st.sidebar.slider("Effective COP", 0.2, 0.8, 0.4, 0.05)

st.sidebar.markdown('<div class="sub-header">Calibration Constants</div>', unsafe_allow_html=True)
C1 = st.sidebar.number_input("C1 (pressure loss)", 0.5, 3.0, 1.0, 0.1)
C2 = st.sidebar.number_input("C2 (angle factor)", 0.5, 5.0, 2.0, 0.1)

geom = KirigamiGeometry(L_flow, W, H_face, theta, phi, d_h, N_rows)
fin_geom = FinGeometry(A_base, A_fin, t_fin, L_f)
peltier = PeltierParams(P_elec, COP_eff)

if st.sidebar.button("Calculate Performance", type="primary"):
    st.session_state.results = run_dashboard_calculation(
        T_inf, RH, Q_flow, geom, fin_geom, peltier, C1, C2
    )

if 'results' in st.session_state:
    results = st.session_state.results
    
    st.markdown('<div class="sub-header">Key Performance Indicators</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Pressure Drop", 
            f"{results['pressure_drop_pa']:.1f} Pa",
            delta=None,
            help="Pressure loss across the kirigami structure"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        delta_T = results['surface_temp_c'] - T_inf
        st.metric(
            "Surface Temperature", 
            f"{results['surface_temp_c']:.1f} °C",
            delta=f"{delta_T:.1f} °C",
            help="Temperature of the cooling surface"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Water Production", 
            f"{results['water_production_ml_hr']:.2f} mL/hr",
            help="Hourly water production rate"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        feasible_text = "PASS" if results['feasible'] else "FAIL"
        feasible_color = "normal" if results['feasible'] else "inverse"
        st.metric(
            "Feasibility Status", 
            feasible_text,
            delta=f"{results['cooling_margin_percent']:.1f}% margin",
            help="Whether cooling capacity meets thermal load"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Humidity Information
    st.markdown('<div class="sub-header">Humidity Information</div>', unsafe_allow_html=True)
    humidity_cols = st.columns(3)
    
    with humidity_cols[0]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Relative Humidity", f"{RH*100:.1f} %")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with humidity_cols[1]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Humidity Ratio", f"{results['ambient_humidity_ratio_g_per_kg']:.2f} g/kg")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with humidity_cols[2]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Dew Point", f"{results['dew_point_c']:.1f} °C")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Temperature Profile</div>', unsafe_allow_html=True)
        
        temps = [T_inf, results['dew_point_c'], results['surface_temp_c']]
        labels = ['Ambient Temperature', 'Dew Point Temperature', 'Surface Temperature']
        colors = ['#1e40af', '#0369a1', '#1d4ed8']
        
        fig = go.Figure()
        
        for i, (temp, label, color) in enumerate(zip(temps, labels, colors)):
            fig.add_trace(go.Bar(
                x=[label],
                y=[temp],
                name=label,
                marker_color=color,
                text=[f"{temp:.1f}°C"],
                textposition='outside',
                marker_line_color='white',
                marker_line_width=1,
                hovertemplate=f"<b>{label}</b><br>Temperature: {temp:.1f}°C<extra></extra>"
            ))
        
        fig.update_layout(
            height=450,
            template="plotly_white",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12),
            margin=dict(t=30, b=80, l=50, r=30),
            xaxis=dict(tickangle=45)
        )
        
        st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Energy Balance Diagram</div>', unsafe_allow_html=True)
        
        # Create Sankey diagram with clear labels
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=20,
                thickness=25,
                line=dict(color="white", width=1),
                label=[
                    "Electrical Input", 
                    "Cooling Capacity", 
                    "Heat Rejection", 
                    "Sensible Heat Load", 
                    "Latent Heat Load"
                ],
                color=["#1e40af", "#0369a1", "#1d4ed8", "#0ea5e9", "#3b82f6"]
            ),
            link=dict(
                source=[0, 0, 1, 1],
                target=[1, 2, 3, 4],
                value=[
                    results['cooling_capacity_w'],
                    P_elec,
                    results['sensible_load_w'],
                    results['latent_load_w']
                ],
                label=[
                    f"Cooling: {results['cooling_capacity_w']:.2f} W",
                    f"Heat Rej: {P_elec:.2f} W",
                    f"Sensible: {results['sensible_load_w']:.2f} W",
                    f"Latent: {results['latent_load_w']:.2f} W"
                ],
                color=["rgba(30, 64, 175, 0.6)", "rgba(29, 78, 216, 0.6)", 
                       "rgba(14, 165, 233, 0.6)", "rgba(59, 130, 246, 0.6)"]
            )
        )])
        
        fig.update_layout(
            height=450,
            font=dict(size=12, color="#333333"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=30, b=30, l=30, r=30)
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Add energy balance summary
        st.markdown("**Energy Balance Summary:**")
        energy_data = {
            "Parameter": ["Electrical Input", "Cooling Capacity", "Heat Rejection", 
                         "Sensible Load", "Latent Load", "Total Load"],
            "Value (W)": [P_elec, results['cooling_capacity_w'], P_elec,
                         results['sensible_load_w'], results['latent_load_w'], 
                         results['total_load_w']]
        }
        energy_df = pd.DataFrame(energy_data)
        st.dataframe(energy_df, width='stretch', hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.expander("Thermal Performance Details", expanded=False):
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Heat Transfer Coefficient", f"{results['heat_transfer_coeff']:.2f} W/m²·K")
            st.metric("Nusselt Number", f"{results['nusselt_number']:.2f}")
            st.metric("Fin Efficiency", f"{results['fin_efficiency']*100:.1f}%")
        
        with col2:
            st.metric("Effective Area", f"{results['effective_area_m2']*10000:.1f} cm²")
            st.metric("Condensation Rate", f"{results['condensation_rate_kg_s']*1000:.4f} g/s")
            st.metric("Cooling Capacity", f"{results['cooling_capacity_w']:.2f} W")
        
        with col3:
            st.metric("Total Load", f"{results['total_load_w']:.2f} W")
            st.metric("Cooling Margin", f"{results['cooling_margin_w']:.2f} W")
            st.metric("Specific Humidity", f"{results['specific_humidity_g_per_kg']:.2f} g/kg")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("Flow Characteristics Details", expanded=False):
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Reynolds Number", f"{results['reynolds_number']:.0f}")
            flow_regime = "Laminar" if results['reynolds_number'] < 2300 else "Turbulent"
            st.metric("Flow Regime", flow_regime)
        
        with col2:
            st.metric("Air Velocity", f"{results['air_velocity_m_s']:.2f} m/s")
            st.metric("Air Mass Flow", f"{results['air_mass_flow_kg_s']*1000:.2f} g/s")
        
        with col3:
            A_face = geom.W * geom.H_face
            A_open = geom.phi * A_face
            st.metric("Face Area", f"{A_face*10000:.1f} cm²")
            st.metric("Open Area", f"{A_open*10000:.1f} cm²")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Performance Analysis
    st.markdown('<div class="sub-header">Performance Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        efficiency = (results['water_production_ml_hr'] / P_elec) if P_elec > 0 else 0
        st.metric("Water Production Efficiency", f"{efficiency:.3f} mL/hr·W")
    
    with col2:
        utilization = (results['total_load_w'] / results['cooling_capacity_w'] * 100) if results['cooling_capacity_w'] > 0 else 0
        st.metric("Cooling Capacity Utilization", f"{utilization:.1f}%")
    
    with col3:
        performance_ratio = (results['water_production_ml_hr'] / results['total_load_w']) if results['total_load_w'] > 0 else 0
        st.metric("Performance Ratio", f"{performance_ratio:.3f} mL/hr·W")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display feasibility message with styling
    if results['feasible']:
        st.markdown(f"""
        <div class="success-box">
            <h4 style="color: #0369a1; margin-top: 0;">System Feasible</h4>
            <p>The cooling capacity is sufficient to meet the thermal load with a margin of {results['cooling_margin_percent']:.1f}% ({results['cooling_margin_w']:.2f} W).</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="warning-box">
            <h4 style="color: #d97706; margin-top: 0;">System Infeasible</h4>
            <p>The cooling capacity is insufficient by {-results['cooling_margin_w']:.2f} W. Consider:</p>
            <ul>
                <li>Increasing electrical power input</li>
                <li>Improving heat transfer coefficient</li>
                <li>Increasing effective area</li>
                <li>Reducing air flow rate</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Export Results
    st.markdown('<div class="sub-header">Export Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    
    results_df = pd.DataFrame([{
        'Timestamp': pd.Timestamp.now(),
        'Ambient_Temp_C': T_inf,
        'Relative_Humidity_%': RH * 100,
        'Humidity_Ratio_g_per_kg': results['ambient_humidity_ratio_g_per_kg'],
        'Dew_Point_C': results['dew_point_c'],
        'Flow_Rate_m3_s': Q_flow,
        'Surface_Temp_C': results['surface_temp_c'],
        'Water_Production_mL_hr': results['water_production_ml_hr'],
        'Pressure_Drop_Pa': results['pressure_drop_pa'],
        'Feasible': results['feasible'],
        'Cooling_Margin_W': results['cooling_margin_w'],
        'Cooling_Margin_%': results['cooling_margin_percent'],
        'Reynolds_Number': results['reynolds_number'],
        'Heat_Transfer_Coeff_W_m2K': results['heat_transfer_coeff'],
        'Fin_Efficiency_%': results['fin_efficiency'] * 100,
        'Cooling_Capacity_W': results['cooling_capacity_w'],
        'Total_Load_W': results['total_load_w'],
        'Sensible_Load_W': results['sensible_load_w'],
        'Latent_Load_W': results['latent_load_w'],
        'Electrical_Power_W': P_elec,
        'Effective_COP': COP_eff,
        'Base_Area_cm2': A_base * 10000,
        'Base_Input_Method': base_input_method
    }])
    
    csv = results_df.to_csv(index=False)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"kirigami_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_csv"
        )
    
    with col2:
        st.info(f"Base Area: {A_base*10000:.1f} cm²")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### Welcome to Kirigami Air-to-Water Converter Dashboard
    
    Configure your system parameters in the sidebar and click **"Calculate Performance"** 
    to analyze the air-to-water conversion efficiency.
    
    **Key Features:**
    1. **Flexible Base Area Input**: Choose between manual input or image-based analysis
    2. **Comprehensive Thermal Analysis**: Detailed energy balance and heat transfer calculations
    3. **Humidity Analysis**: Multiple humidity metrics including dew point and humidity ratio
    4. **Flow Characteristics**: Reynolds number, pressure drop, and flow regime analysis
    5. **Performance Metrics**: Water production rate and system efficiency calculations
    
    **Instructions:**
    1. Set environmental conditions (temperature, humidity, flow rate)
    2. Configure kirigami geometry parameters
    3. Choose base area input method (manual or image analysis)
    4. Set Peltier module parameters
    5. Click "Calculate Performance" to run the analysis
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Based on calculation framework by Hitendra Vaishnav | Sustainable Water Harvesting Technology")