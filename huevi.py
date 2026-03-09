import streamlit as st
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

#####################################################
#### BARRAS DE PARÁMETROS Y RESOLUCIÓN DEL MODELO ###
#####################################################

st.set_page_config(page_title="Simulador Heckscher-Ohlin", layout="wide")

# Funciones Matemáticas 
def requerimiento_factor(w1, w2, alfa, A):
    w1 = max(w1, 1e-12)
    w2 = max(w2, 1e-12)
    return (alfa/(1-alfa))**(1-alfa) * w1**(alfa-1) * w2**(1-alfa) / A

def demanda(p, I, beta):
    return beta * I / p

def beneficio_cero(vars, p1, a1, a2, p2=1.0):
    w, r = vars
    R1 = p1 - requerimiento_factor(w, r, a1, 1.0)*w - requerimiento_factor(r, w, 1-a1, 1.0)*r
    R2 = p2 - requerimiento_factor(w, r, a2, 1.0)*w - requerimiento_factor(r, w, 1-a2, 1.0)*r
    return [R1, R2]

def mercado_factores(vars, w, r, L_val, K_val, a1, a2):
    Y1, Y2 = vars
    R1 = L_val - requerimiento_factor(w, r, a1, 1.0)*Y1 - requerimiento_factor(w, r, a2, 1.0)*Y2
    R2 = K_val - requerimiento_factor(r, w, 1-a1, 1.0)*Y1 - requerimiento_factor(r, w, 1-a2, 1.0)*Y2
    return [R1, R2]

def get_ppf(L_val, K_val, a1, a2):
    c1, c2 = a1/(1-a1), a2/(1-a2)
    l1_range = np.linspace(0.1, L_val-0.1, 100)
    k1_range = (c2 * K_val * l1_range) / (c1 * L_val + (c2 - c1) * l1_range)
    q1 = (l1_range**a1) * (k1_range**(1-a1))
    q2 = ((L_val-l1_range)**a2) * ((K_val-k1_range)**(1-a2))
    return q1, q2

# Barras de parámetros
input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    with st.expander("País Nacional ", expanded=True):
        L = st.slider("Trabajo (L)", 50.0, 500.0, 200.0)
        K = st.slider("Capital (K)", 50.0, 500.0, 100.0)

with input_col2:
    with st.expander("País Extranjero", expanded=True):
        L_f = st.slider("Trabajo (L*)", 50.0, 500.0, 100.0)
        K_f = st.slider("Capital (K*)", 50.0, 500.0, 200.0)

with input_col3:
    with st.expander("Parámetros Globales", expanded=False):
        beta = st.slider("Preferencia por el bien 1", 0.1, 0.9, 0.4)
        alfa1 = st.slider("Intensidad L en Bien 1 (α1)", 0.1, 0.9, 0.9)
        alfa2 = st.slider("Intensidad L en Bien 2 (α2)", 0.1, 0.9, 0.1)
        p2 = 1.0

# Funciones de solución del modelo 

def exceso_demanda_global(p1):
    p1 = float(np.atleast_1d(p1)[0])
    w_h, r_h = fsolve(beneficio_cero, (1.0, 1.0), args=(p1, alfa1, alfa2))
    w_ex, r_ex = fsolve(beneficio_cero, (1.0, 1.0), args=(p1, alfa1, alfa2))
    Y1_h, Y2_h = fsolve(mercado_factores, (100.0, 100.0), args=(w_h, r_h, L, K, alfa1, alfa2))
    Y1_ex, Y2_ex = fsolve(mercado_factores, (100.0, 100.0), args=(w_ex, r_ex, L_f, K_f, alfa1, alfa2))
    I_h = w_h * L + r_h * K
    I_ex = w_ex * L_f + r_ex * K_f
    return demanda(p1, I_h, beta) + demanda(p1, I_ex, beta) - Y1_h - Y1_ex

def exceso_demanda_local(p1):
    p1 = float(np.atleast_1d(p1)[0])
    w_h, r_h = fsolve(beneficio_cero, (1.0, 1.0), args=(p1, alfa1, alfa2))
    Y1_h, Y2_h = fsolve(mercado_factores, (100.0, 100.0), args=(w_h, r_h, L, K, alfa1, alfa2))
    I_h = w_h * L + r_h * K
    return demanda(p1, I_h, beta) - Y1_h

def exceso_demanda_extranjero(p1):
    p1 = float(np.atleast_1d(p1)[0])
    w_h, r_h = fsolve(beneficio_cero, (1.0, 1.0), args=(p1, alfa1, alfa2))
    Y1_h, Y2_h = fsolve(mercado_factores, (100.0, 100.0), args=(w_h, r_h, L_f, K_f, alfa1, alfa2))
    I_h = w_h * L_f + r_h * K_f 
    return demanda(p1, I_h, beta) - Y1_h

# calculos de cada economia por separado para hacer la comparacion en las ganancias del comercio
try:
    p1_1_sol = fsolve(exceso_demanda_local, 1.0)[0]
    w_1, r_1 = fsolve(beneficio_cero, (1.0, 1.0), args=(p1_1_sol, alfa1, alfa2))
    Y1_1, Y2_1 = fsolve(mercado_factores, (100.0, 100.0), args=(w_1, r_1, L, K, alfa1, alfa2))
    I_1 = w_1 * L + r_1 * K
    C1_1, C2_1 = demanda(p1_1_sol, I_1, beta), demanda(p2, I_1, 1-beta)
    al1_1 = requerimiento_factor(w_1, r_1, alfa1, 1); al2_1 = requerimiento_factor(w_1, r_1, alfa2, 1)
    ak1_1 = requerimiento_factor(r_1, w_1, 1-alfa1, 1); ak2_1 = requerimiento_factor(r_1, w_1, 1-alfa2, 1)
    L1_1_sol = al1_1 * Y1_1; L2_1_sol = al2_1 * Y2_1
    K1_1_sol = ak1_1 * Y1_1; K2_1_sol = ak2_1 * Y2_1

except Exception as e:
    st.warning("⚠️ El modelo no pudo converger para el país nacional en autarquía. Esto suele ocurrir cuando las dotaciones son muy extremas o las intensidades de factores son idénticas.")

try:
    p1_2_sol = fsolve(exceso_demanda_extranjero, 1.0)[0]
    w_2, r_2 = fsolve(beneficio_cero, (1.0, 1.0), args=(p1_2_sol, alfa1, alfa2))
    Y1_2, Y2_2 = fsolve(mercado_factores, (100.0, 100.0), args=(w_2, r_2, L_f, K_f, alfa1, alfa2))
    I_2 = w_2 * L_f + r_2 * K_f
    C1_2, C2_2 = demanda(p1_2_sol, I_2, beta), demanda(p2, I_2, 1-beta)
    al1_2 = requerimiento_factor(w_2, r_2, alfa1, 1); al2_2 = requerimiento_factor(w_2, r_2, alfa2, 1)
    ak1_2 = requerimiento_factor(r_2, w_2, 1-alfa1, 1); ak2_2 = requerimiento_factor(r_2, w_2, 1-alfa2, 1)
    L1_2_sol = al1_2 * Y1_2; L2_2_sol = al2_2 * Y2_2
    K1_2_sol = ak1_2 * Y1_2; K2_2_sol = ak2_2 * Y2_2
except Exception as e:
    st.warning("⚠️ El modelo no pudo converger para el país extranjero en autarquía. Esto suele ocurrir cuando las dotaciones son muy extremas o las intensidades de factores son idénticas.")


# Resolución del modelo

try:
    p1_sol = fsolve(exceso_demanda_global, 1.0)[0]
    
    
    w, r = fsolve(beneficio_cero, (1.0, 1.0), args=(p1_sol, alfa1, alfa2))
    w_f, r_f = fsolve(beneficio_cero, (1.0, 1.0), args=(p1_sol, alfa1, alfa2))
    Y1, Y2 = fsolve(mercado_factores, (100.0, 100.0), args=(w, r, L, K, alfa1, alfa2))
    Y1_f, Y2_f = fsolve(mercado_factores, (100.0, 100.0), args=(w_f, r_f, L_f, K_f, alfa1, alfa2))
    I_h, I_ex = w * L + r * K, w_f * L_f + r_f * K_f
    C1, C2 = demanda(p1_sol, I_h, beta), demanda(p2, I_h, 1-beta)
    C1_f, C2_f = demanda(p1_sol, I_ex, beta), demanda(p2, I_ex, 1-beta)

    al1 = requerimiento_factor(w, r, alfa1, 1); al2 = requerimiento_factor(w, r, alfa2, 1)
    ak1 = requerimiento_factor(r, w, 1-alfa1, 1); ak2 = requerimiento_factor(r, w, 1-alfa2, 1)
    al1_f = requerimiento_factor(w_f, r_f, alfa1, 1); al2_f = requerimiento_factor(w_f, r_f, alfa2, 1)
    ak1_f = requerimiento_factor(r_f, w_f, 1-alfa1, 1); ak2_f = requerimiento_factor(r_f, w_f, 1-alfa2, 1)

    L1_sol = al1 * Y1; L2_sol = al2 * Y2
    K1_sol = ak1 * Y1; K2_sol = ak2 * Y2
    L1_sol_f = al1_f * Y1_f; L2_sol_f = al2_f * Y2_f
    K1_sol_f = ak1_f * Y1_f; K2_sol_f = ak2_f * Y2_f

    exp1_h = Y1 - C1  # Si es +, Nacional exporta Bien 1
    exp2_h = Y2 - C2  # Si es +, Nacional exporta Bien 2

    cambio_w = w - w_1
    cambio_r = r - r_1
    cambio_w_f = w_f - w_2
    cambio_r_f = r_f - r_2

    ###########################################################################################
    # Las graficas  y las tablas se encuentran dentro del bloque try de resolución del modelo#
    ######################################################+###################################

    # Layout de resultados (2/3 gráfico, 1/3 métricas)
    res_col1, res_col2 = st.columns([2, 1])

    # gráfica



    with res_col1:
       
        st.subheader("Frontera de Posibilidades de Producción (FPP)")

        escenario = st.radio(
                "Selecciona el escenario:",
                ["Autarquía", "Comercio"],
                horizontal=True
            )

        fig, ax = plt.subplots(figsize=(10, 6))
        q1_n, q2_n = get_ppf(L, K, alfa1, alfa2)
        q1_f, q2_f = get_ppf(L_f, K_f, alfa1, alfa2)
        
        ax.plot(q1_n, q2_n, label="FPP Nacional", color="#1f77b4", lw=2)
        ax.plot(q1_f, q2_f, label="FPP Extranjero", color="#ff7f0e", lw=2)
        if escenario == "Comercio":
            # Líneas de precios internacionales (Presupuesto)
            lim = 1.1 * np.max(np.concatenate([q1_n, q2_n, q1_f, q2_f]))
            x = np.linspace(0, lim, 300)
            ax.plot(x, Y2 - p1_sol*(x - Y1), lw=1, ls="--", color="gray", alpha=0.5)
            ax.plot(x, Y2_f - p1_sol*(x - Y1_f), lw=1, ls="--", color="gray", alpha=0.5)
            
            # Puntos de producción y consumo en comercio
            ax.scatter([Y1, C1, Y1_f, C1_f], [Y2, C2, Y2_f, C2_f], color=["#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e"], s=50)
            ax.annotate("Y nac", (Y1, Y2), xytext=(5,5), textcoords='offset points')
            ax.annotate("C nac", (C1, C2), xytext=(5,5), textcoords='offset points')
            ax.annotate("Y ext", (Y1_f, Y2_f), xytext=(5,5), textcoords='offset points')
            ax.annotate("C ext", (C1_f, C2_f), xytext=(5,5), textcoords='offset points')
    
        else:

            lim = 1.1 * np.max(np.concatenate([q1_n, q2_n, q1_f, q2_f]))
            x = np.linspace(0, lim, 300)
            ax.plot(x, Y2_1 - p1_1_sol*(x - Y1_1), lw=1, ls="--", color="gray", alpha=0.5)
            ax.plot(x, Y2_2 - p1_2_sol*(x - Y1_2), lw=1, ls="--", color="gray", alpha=0.5)
            
            # Lógica para Autarquía: El consumo es igual a la producción (C = Y)
            # Aquí deberías usar tus variables de equilibrio de autarquía si las tienes
            ax.scatter([Y1_1, Y1_2], [Y2_1, Y2_2], color=["#1f77b4", "#ff7f0e"], s=50)
            ax.annotate("Y nac", (Y1_1, Y2_1), xytext=(5,5), textcoords='offset points')
            ax.annotate("Y ext", (Y1_2, Y2_2), xytext=(5,5), textcoords='offset points')
            
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)

        ax.set_xlabel("Bien 1")
        ax.set_ylabel("Bien 2")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)

    
    # tablas de resultados

    with res_col2:
        st.subheader("Análisis de Equilibrio", help="Se está tomando el precio del bien 2 como numerario.")

        st.metric("Precio relativo del bien 1", f"{p1_sol:.2f}")
    
        
        tab_h, tab_f, tab_c = st.tabs(["🏠 Nacional", "🌍 Extranjero", "📈 Ganancias del Comercio"])
        
        with tab_h:
            u_col1, u_col2 = st.columns(2)
            u_col1.metric("Salario", f"{w:.2f}", delta=round(cambio_w, 2))
            u_col2.metric("Renta", f"{r:.2f}", delta=round(cambio_r, 2))

            # Tabla simple de Producción vs Consumo
            st.dataframe(pd.DataFrame({
                "Bien": ["1", "2"],
                "Producción": [f"{Y1:.1f}", f"{Y2:.1f}"],
                "Consumo": [f"{C1:.1f}", f"{C2:.1f}"],
                "Flujo Neto": [f"{exp1_h:+.1f}", f"{exp2_h:+.1f}"]
            }), hide_index=True)

            st.write("Asignación de los factores")
            st.dataframe(pd.DataFrame({
                "Bien": ["1", "2"],
                "Trabajo": [f"{L1_sol:.1f}", f"{L2_sol:.1f}"],
                "Capital": [f"{K1_sol:.1f}", f"{K2_sol:.1f}"]
            }), hide_index=True)

        with tab_f:
            u_col1, u_col2 = st.columns(2)
            u_col1.metric("Salario", f"{w_f:.2f}", delta=round(cambio_w_f, 2))
            u_col2.metric("Renta", f"{r_f:.2f}", delta=round(cambio_r_f, 2))
            
            st.dataframe(pd.DataFrame({
                "Bien": ["1", "2"],
                "Producción": [f"{Y1_f:.1f}", f"{Y2_f:.1f}"],
                "Consumo": [f"{C1_f:.1f}", f"{C2_f:.1f}"],
                "Flujo Neto": [f"{-exp1_h:+.1f}", f"{-exp2_h:+.1f}"]
            }), hide_index=True)

            st.write("Asignación de los factores")
            st.dataframe(pd.DataFrame({
                "Bien": ["1", "2"],
                "Trabajo": [f"{L1_sol_f:.1f}", f"{L2_sol_f:.1f}"],
                "Capital": [f"{K1_sol_f:.1f}", f"{K2_sol_f:.1f}"]
            }), hide_index=True)


        with tab_c:
            st.write("Ganancias en producción")
            st.dataframe(pd.DataFrame({
                "Bien": ["1", "2"],
                "Nacional": [f"{Y1-Y1_1:+.1f}", f"{Y2-Y2_1:+.1f}"],
                "Extranjero": [f"{Y1_f-Y1_2:+.1f}", f"{Y2_f-Y2_2:+.1f}"],
                "Global": [
                    f"{(Y1-Y1_1) + (Y1_f-Y1_2):+.1f}",
                    f"{(Y2-Y2_1) + (Y2_f-Y2_2):+.1f}"
                ]
            }), hide_index=True)

            st.write("Ganancias en consumo")
            st.dataframe(pd.DataFrame({
                "Bien": ["1", "2"],
                "Nacional": [f"{C1-C1_1:+.1f}", f"{C2-C2_1:+.1f}"],
                "Extranjero": [f"{C1_f-C1_2:+.1f}", f"{C2_f-C2_2:+.1f}"],
                "Global": [
                    f"{(C1-C1_1) + (C1_f-C1_2):+.1f}",
                    f"{(C2-C2_1) + (C2_f-C2_2):+.1f}"
                ]
            }), hide_index=True)

except Exception as e:
    st.warning(f"⚠️ El modelo no pudo converger con estos parámetros. Esto suele ocurrir cuando las dotaciones son muy extremas o las intensidades de factores son idénticas: {e}")