import streamlit as st
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Simulador Heckscher-Ohlin", layout="wide")

st.title("📈 Modelo Heckscher-Ohlin con Comercio")
st.markdown("""
Esta aplicación simula el equilibrio de comercio entre dos países con diferentes dotaciones de factores (Trabajo $L$ y Capital $K$).
""")

# --- Sidebar para Parámetros ---
st.sidebar.header("Configuración de Parámetros")

with st.sidebar.expander("País Nacional (Home)", expanded=True):
    L = st.slider("Trabajo (L)", 50.0, 500.0, 200.0)
    K = st.slider("Capital (K)", 50.0, 500.0, 100.0)
    alfa1 = st.slider("Intensidad L en Bien 1 (α1)", 0.1, 0.9, 0.9)
    alfa2 = st.slider("Intensidad L en Bien 2 (α2)", 0.1, 0.9, 0.1)

with st.sidebar.expander("País Extranjero (Foreign)", expanded=True):
    L_f = st.slider("Trabajo (L*)", 50.0, 500.0, 100.0)
    K_f = st.slider("Capital (K*)", 50.0, 500.0, 200.0)
    alfa1_f = st.slider("Intensidad L* en Bien 1", 0.1, 0.9, 0.9)
    alfa2_f = st.slider("Intensidad L* en Bien 2", 0.1, 0.9, 0.1)

# Parámetros fijos o compartidos
beta = 0.4
p2 = 1.0
A1 = A2 = A1_f = A2_f = 1.0

# --- Lógica del Modelo ---

def requerimiento_factor(w1, w2, alfa, A):
    w1 = max(w1, 1e-12)
    w2 = max(w2, 1e-12)
    return (alfa/(1-alfa))**(1-alfa) * w1**(alfa-1) * w2**(1-alfa) / A

def demanda(p, I, beta):
    return beta * I / p

def beneficio_cero(vars, p1, a1, a2):
    w, r = vars
    R1 = p1 - requerimiento_factor(w, r, a1, 1.0)*w - requerimiento_factor(r, w, 1-a1, 1.0)*r
    R2 = p2 - requerimiento_factor(w, r, a2, 1.0)*w - requerimiento_factor(r, w, 1-a2, 1.0)*r
    return [R1, R2]

def mercado_factores(vars, w, r, L_val, K_val, a1, a2):
    Y1, Y2 = vars
    R1 = L_val - requerimiento_factor(w, r, a1, 1.0)*Y1 - requerimiento_factor(w, r, a2, 1.0)*Y2
    R2 = K_val - requerimiento_factor(r, w, 1-a1, 1.0)*Y1 - requerimiento_factor(r, w, 1-a2, 1.0)*Y2
    return [R1, R2]

def exceso_demanda(p1):
    p1 = float(np.atleast_1d(p1)[0])
    w_h, r_h = fsolve(beneficio_cero, (1.0, 1.0), args=(p1, alfa1, alfa2))
    w_ex, r_ex = fsolve(beneficio_cero, (1.0, 1.0), args=(p1, alfa1_f, alfa2_f))
    Y1_h, Y2_h = fsolve(mercado_factores, (100.0, 100.0), args=(w_h, r_h, L, K, alfa1, alfa2))
    Y1_ex, Y2_ex = fsolve(mercado_factores, (100.0, 100.0), args=(w_ex, r_ex, L_f, K_f, alfa1_f, alfa2_f))
    
    I_h = w_h * L + r_h * K
    I_ex = w_ex * L_f + r_ex * K_f
    return demanda(p1, I_h, beta) + demanda(p1, I_ex, beta) - Y1_h - Y1_ex

# Resolución
try:
    p1_sol = fsolve(exceso_demanda, 1.0)[0]
    
    # Cálculos finales
    w, r = fsolve(beneficio_cero, (1.0, 1.0), args=(p1_sol, alfa1, alfa2))
    w_f, r_f = fsolve(beneficio_cero, (1.0, 1.0), args=(p1_sol, alfa1_f, alfa2_f))
    Y1, Y2 = fsolve(mercado_factores, (100.0, 100.0), args=(w, r, L, K, alfa1, alfa2))
    Y1_f, Y2_f = fsolve(mercado_factores, (100.0, 100.0), args=(w_f, r_f, L_f, K_f, alfa1_f, alfa2_f))
    I_h, I_ex = w * L + r * K, w_f * L_f + r_f * K_f
    C1, C2 = demanda(p1_sol, I_h, beta), demanda(p2, I_h, 1-beta)
    C1_f, C2_f = demanda(p1_sol, I_ex, beta), demanda(p2, I_ex, 1-beta)

    # --- Interfaz Principal ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Fronteras de Posibilidad de Producción (FPP)")
        
        def get_ppf(L_val, K_val, a1, a2):
            c1, c2 = a1/(1-a1), a2/(1-a2)
            l1_range = np.linspace(0.1, L_val-0.1, 100)
            k1_range = (c2 * K_val * l1_range) / (c1 * L_val + (c2 - c1) * l1_range)
            q1 = (l1_range**a1) * (k1_range**(1-a1))
            q2 = ((L_val-l1_range)**a2) * ((K_val-k1_range)**1-a2)
            return q1, q2

        fig, ax = plt.subplots(figsize=(10, 7))
        q1_n, q2_n = get_ppf(L, K, alfa1, alfa2)
        q1_f, q2_f = get_ppf(L_f, K_f, alfa1_f, alfa2_f)
        
        ax.plot(q1_n, q2_n, label="FPP Nacional", color="blue", lw=2)
        ax.plot(q1_f, q2_f, label="FPP Extranjero", color="red", lw=2)
        ax.scatter([Y1, C1], [Y2, C2], color="blue")
        ax.scatter([Y1_f, C1_f], [Y2_f, C2_f], color="red")
        
        ax.set_xlabel("Bien 1")
        ax.set_ylabel("Bien 2")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with col2:
        st.subheader("Resultados Globales")
        st.metric("Precio Relativo Equil. (p1/p2)", f"{p1_sol:.3f}")
        
        tab1, tab2 = st.tabs(["Nacional", "Extranjero"])
        with tab1:
            st.write(f"**Producción:** ({Y1:.1f}, {Y2:.1f})")
            st.write(f"**Consumo:** ({C1:.1f}, {C2:.1f})")
            st.write(f"**Salario (w):** {w:.2f}")
            st.write(f"**Renta Cap (r):** {r:.2f}")
        with tab2:
            st.write(f"**Producción:** ({Y1_f:.1f}, {Y2_f:.1f})")
            st.write(f"**Consumo:** ({C1_f:.1f}, {C2_f:.1f})")
            st.write(f"**Salario (w*):** {w_f:.2f}")
            st.write(f"**Renta Cap (r*):** {r_f:.2f}")

except Exception as e:
    st.error(f"Error en la convergencia del modelo: {e}. Intenta ajustar los parámetros.")