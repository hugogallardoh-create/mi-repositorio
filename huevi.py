import streamlit as st
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import root_scalar
from scipy.optimize import minimize_scalar
import funciones as f

st.set_page_config(page_title="Simulador Heckscher-Ohlin", layout="wide", initial_sidebar_state="collapsed")

####################################
# --- SIDEBAR (BARRA LATERAL) ---
###################################

with st.sidebar:
    st.header("Configuración de Parámetros")
    
    st.subheader("Preferencias Globales")
    modelo_util = st.selectbox(
        "Elige una función de utilidad:",
        ["Cobb-Douglas", "Lineal", "Leontief", "CES", "Cuasi-lineal", "Stone-Gery"]
    )
    
    # Lógica condicional para la Parte 1
    if modelo_util == "Cobb-Douglas":
        st.latex(r"U(x, y) =  x^\alpha  y^{(1-\alpha)}")
        alpha = st.slider("Valor de Alfa (α):", 0.1, 0.95, 0.4)
        st.write('Para la función de utilidad Cobb Douglas, que el grado de homogeneidad sea distinto a uno no posee ningún efecto.')
    elif modelo_util == "Lineal":
        st.latex(r"U(x, y) = \alpha x + \beta y")
        alpha = st.slider("Valor de Alfa (α):", 0.1, 10.0, 0.3)
        beta = st.slider("Valor de Beta (β):", 0.1, 10.0, 0.7)
    elif modelo_util == "Leontief":
        st.latex(r"U(x, y) = \min\{\alpha x, \beta y\}")
        alpha = st.slider("Valor de Alfa (α):", 0.1, 10.0, 0.3)
        beta = st.slider("Valor de Beta (β):", 0.1, 10.0, 0.7)
    elif modelo_util == "CES":
        st.latex(r"U(x, y) = \left[ \delta_1 x^{\frac{\sigma-1}{\sigma}} + delta_2 y^{\frac{\sigma-1}{\sigma}} \right]^{1/{\frac{\sigma}{\sigma-1}}}")
        delta1 = st.slider("Valor de delta_l (d):", 0.1, 1.0, 0.9)
        delta2 = st.slider("Valor de delta_k (d):", 0.1, 1.0, 0.1)
        sigma = st.slider("Valor de sigma (s):", 0.1, 3.0, 0.9)
    elif modelo_util == "Cuasi-lineal":
        st.latex(r"U(x, y) = \alpha \ln(x) + y")
        alpha = st.slider("Valor de Alfa (α):", 0.1, 30.0, 10.0)
    else: #stone gery#
        st.latex(r"U(x, y) = (x - \gamma_x)^\alpha \cdot (y - \gamma_y)^(1-\alpha)")
        alpha = st.slider("Valor de Alfa (α):", 0.1, 1.0, 0.3)
        gamma_x = st.slider("Valor de Gamma X (γx):", 0.0, 50.0, 5.0)
        gamma_y = st.slider("Valor de Gamma Y (γy):", 0.0, 50.0, 5.0)

    st.divider() # Línea divisoria

    # --- PARTE 2: VALORACIÓN FINANCIERA ---
    st.subheader("Tecnología Global")
    modelo_prod = st.selectbox(
        "Elige una función de producción:",
        ["Cobb-Douglas", "CES"]
    )
    
    # Lógica condicional para la Parte 1
    if modelo_prod == "Cobb-Douglas":
        st.latex(r"Y = A  L^a  K^{(1-a)}")
        st.write('Sector del bien 1:')
        A1 = st.slider("Valor de A_1:", 0.1, 10.0, 1.0)
        alpha1_p = st.slider("Valor de Alfa (a_1):", 0.1, 0.95, 0.9)
        st.write('Sector del bien 2:')
        A2 = st.slider("Valor de A_2:", 0.1, 10.0, 1.0)
        alpha2_p = st.slider("Valor de Alfa (a_2):", 0.1, 0.95, 0.1)
        st.write('Para la función de producción Cobb Douglas, que el grado de homogeneidad sea distinto a uno rompe el supuesto de rendimientos constantes a escala.')
    else:
        st.latex(r"Y = A \left[ d_1 L^{\frac{s-1}{s}} + d_2 K^{\frac{s-1}{s}} \right]^{\frac{s}{s-1}}")
        st.write('Sector del bien 1:')
        A1 = st.slider("Valor de A_1:", 0.1, 10.0, 1.0)
        delta11_p = st.slider("Valor de delta_l (d):", 0.1, 1.0, 0.9)
        delta21_p = st.slider("Valor de delta_k (d):", 0.1, 1.0, 0.1)
        sigma1_p = st.slider("Valor de sigma (s):", 0.1, 3.0, 0.9)
        st.write('Sector del bien 2:')
        A2 = st.slider("Valor de A_2:", 0.1, 10.0, 1.0)
        delta12_p = st.slider("Valor de delta_l (d): ", 0.1, 1.0, 0.1)
        delta22_p = st.slider("Valor de delta_k (d): ", 0.1, 1.0, 0.9)
        sigma2_p = st.slider("Valor de sigma (s): ", 0.1, 3.0, 0.9)





#####################################################
####  FUNCIONES MATEMATICAS          ###
#####################################################

# Funciones condicionales según modelo_prod (para requerimiento_factor y get_ppf)

tms_opcional = 1e+10

if modelo_prod == "Cobb-Douglas":
    clase = f.cobb_douglas(A1, A2, alpha1_p, alpha2_p)
    a1l = clase.requerimiento_factor_1l
    a2l = clase.requerimiento_factor_2l
    a1k = clase.requerimiento_factor_1k
    a2k = clase.requerimiento_factor_2k
    ppf = clase.get_ppf

else:  # CES
    objeto = f.ces_production(A1, A2, delta11_p, delta21_p, delta12_p, delta22_p, sigma1_p, sigma2_p)
    a1l = objeto.requerimiento_factor_1l
    a2l = objeto.requerimiento_factor_2l
    a1k = objeto.requerimiento_factor_1k
    a2k = objeto.requerimiento_factor_2k
    ppf = objeto.get_ppf

# Funciones condicionales según modelo_util (para demanda)

if modelo_util == "Cobb-Douglas":
    objeto = f.cobb_douglas_utilidad(alpha)
    con1 = objeto.demanda1
    con2 = objeto.demanda2

elif modelo_util == "Lineal":
    objeto = f.utilidad_lineal(alpha, beta)
    con1 = objeto.demanda1
    con2 = objeto.demanda2
    tms_opcional = alpha / beta  

elif modelo_util == "Leontief":
    objeto = f.utilidad_leontief(alpha, beta)
    con1 = objeto.demanda1
    con2 = objeto.demanda2

elif modelo_util == "CES":
    objeto = f.utilidad_ces(delta1, delta2, sigma)
    con1 = objeto.demanda1
    con2 = objeto.demanda2

elif modelo_util == "Cuasi-lineal":
    objeto = f.utilidad_cuasilineal(alpha)
    con1 = objeto.demanda1
    con2 = objeto.demanda2

else:  # Stone-Gery
    objeto = f.utilidad_stonegery(alpha, gamma_x, gamma_y)
    con1 = objeto.demanda1
    con2 = objeto.demanda2


def beneficio_cero(vars, p1):
    w, r = vars
    R1 = p1 - a1l(w, r)*w - a1k(w, r)*r
    R2 = p2 - a2l(w, r)*w - a2k(w, r)*r
    return [R1, R2]

def mercado_factores(vars, w, r, L_val, K_val):
    Y1, Y2 = vars
    R1 = L_val - a1l(w, r)*Y1 - a2l(w, r)*Y2
    R2 = K_val - a1k(w, r)*Y1 - a2k(w, r)*Y2
    return [R1, R2]



#####################################################
#### BARRAS DE PARÁMETROS Y RESOLUCIÓN DEL MODELO ###
#####################################################

# Barras de parámetros
input_col1, input_col2 = st.columns(2)

with input_col1:
    with st.expander("País Nacional ", expanded=True):
        L = st.slider("Trabajo (L)", 50.0, 1000.0, 200.0)
        K = st.slider("Capital (K)", 50.0, 1000.0, 100.0)

with input_col2:
    with st.expander("País Extranjero", expanded=True):
        L_f = st.slider("Trabajo (L*)", 50.0, 1000.0, 100.0)
        K_f = st.slider("Capital (K*)", 50.0, 1000.0, 200.0)

p2 = 1.0

# Funciones de solución del modelo 

def exceso_demanda_global(p1):
    p1 = float(np.atleast_1d(p1)[0])
    w_h, r_h = fsolve(beneficio_cero, (1.0, 1.0), args=(p1))
    w_ex, r_ex = fsolve(beneficio_cero, (1.0, 1.0), args=(p1))
    Y1_h, Y2_h = fsolve(mercado_factores, (100.0, 100.0), args=(w_h, r_h, L, K))
    Y1_ex, Y2_ex = fsolve(mercado_factores, (100.0, 100.0), args=(w_ex, r_ex, L_f, K_f))
    I_h = w_h * L + r_h * K
    I_ex = w_ex * L_f + r_ex * K_f
    R1 = con1(p1, I_h) + con1(p1, I_ex) - Y1_h - Y1_ex
    if p1 == tms_opcional:
        R1 = 0
    return R1

def exceso_demanda_local(p1):
    p1 = float(np.atleast_1d(p1)[0])
    w_h, r_h = fsolve(beneficio_cero, (1.0, 1.0), args=(p1))
    Y1_h, Y2_h = fsolve(mercado_factores, (100.0, 100.0), args=(w_h, r_h, L, K))
    I_h = w_h * L + r_h * K
    R1 = con1(p1, I_h) - Y1_h
    if p1 == tms_opcional:
        R1 = 0
    return R1

def exceso_demanda_extranjero(p1):
    p1 = float(np.atleast_1d(p1)[0])
    w_h, r_h = fsolve(beneficio_cero, (1.0, 1.0), args=(p1))
    Y1_h, Y2_h = fsolve(mercado_factores, (100.0, 100.0), args=(w_h, r_h, L_f, K_f))
    I_h = w_h * L_f + r_h * K_f 
    R1 = con1(p1, I_h) - Y1_h
    if p1 == tms_opcional:
        R1 = 0
    return R1



# calculos de cada economia por separado para hacer la comparacion en las ganancias del comercio
try:
    p1_1_sol = fsolve(exceso_demanda_local, 1.0)[0]
    w_1, r_1 = fsolve(beneficio_cero, (1.0, 1.0), args=(p1_1_sol))
    Y1_1, Y2_1 = fsolve(mercado_factores, (100.0, 100.0), args=(w_1, r_1, L, K))
    I_1 = w_1 * L + r_1 * K
    C1_1, C2_1 = con1(p1_1_sol, I_1), con2(p2, I_1)
    if p1_1_sol == tms_opcional:
        C1_1, C2_1 = Y1_1, Y2_1
    al1_1 = a1l(w_1, r_1); al2_1 = a2l(w_1, r_1)
    ak1_1 = a1k(w_1, r_1); ak2_1 = a2k(w_1, r_1)
    L1_1_sol = al1_1 * Y1_1; L2_1_sol = al2_1 * Y2_1
    K1_1_sol = ak1_1 * Y1_1; K2_1_sol = ak2_1 * Y2_1

except Exception as e:
    st.warning("⚠️ El modelo no pudo converger para el país nacional en autarquía. Esto suele ocurrir cuando las dotaciones son muy extremas o las intensidades de factores son idénticas.")

try:
    p1_2_sol = fsolve(exceso_demanda_extranjero, 1.0)[0]
    w_2, r_2 = fsolve(beneficio_cero, (1.0, 1.0), args=(p1_2_sol))
    Y1_2, Y2_2 = fsolve(mercado_factores, (100.0, 100.0), args=(w_2, r_2, L_f, K_f))
    I_2 = w_2 * L_f + r_2 * K_f
    C1_2, C2_2 = con1(p1_2_sol, I_2), con2(p2, I_2)
    if p1_2_sol == tms_opcional:
        C1_2, C2_2 = Y1_2, Y2_2
    al1_2 = a1l(w_2, r_2); al2_2 = a2l(w_2, r_2)
    ak1_2 = a1k(w_2, r_2); ak2_2 = a2k(w_2, r_2)
    L1_2_sol = al1_2 * Y1_2; L2_2_sol = al2_2 * Y2_2
    K1_2_sol = ak1_2 * Y1_2; K2_2_sol = ak2_2 * Y2_2
except Exception as e:
    st.warning("⚠️ El modelo no pudo converger para el país extranjero en autarquía. Esto suele ocurrir cuando las dotaciones son muy extremas o las intensidades de factores son idénticas.")


# Resolución del modelo

try:
    p1_sol = fsolve(exceso_demanda_global, 1.0)[0]
    
    
    w, r = fsolve(beneficio_cero, (1.0, 1.0), args=(p1_sol))
    w_f, r_f = fsolve(beneficio_cero, (1.0, 1.0), args=(p1_sol))
    Y1, Y2 = fsolve(mercado_factores, (100.0, 100.0), args=(w, r, L, K))
    Y1_f, Y2_f = fsolve(mercado_factores, (100.0, 100.0), args=(w_f, r_f, L_f, K_f))
    I_h, I_ex = w * L + r * K, w_f * L_f + r_f * K_f
    C1, C2 = con1(p1_sol, I_h), con2(p2, I_h)
    C1_f, C2_f = con1(p1_sol, I_ex), con2(p2, I_ex)
    if p1_1_sol == tms_opcional:
        C1, C2 = Y1, Y2
        C1_f, C2_f = Y1_f, Y2_f

    al1 = a1l(w, r); al2 = a2l(w, r)
    ak1 = a1k(w, r); ak2 = a2k(w, r)
    al1_f = a1l(w_f, r_f); al2_f = a2l(w_f, r_f)
    ak1_f = a1k(w_f, r_f); ak2_f = a2k(w_f, r_f)

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
        q1_n, q2_n = ppf(L, K)
        q1_f, q2_f = ppf(L_f, K_f)
        
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

        if escenario == "Comercio":
            st.metric("Precio relativo del bien 1 (p1/p2)", f"{p1_sol:.2f}")
        
            tab_h, tab_f, tab_c = st.tabs(["🏠 Nacional", "🌍 Extranjero", "📈 Ganancias del Comercio"])
            
            with tab_h:
                u_col1, u_col2 = st.columns(2)
                u_col1.metric("Salario (w/p2)", f"{w:.2f}", delta=round(cambio_w, 2))
                u_col2.metric("Renta (r/p2)", f"{r:.2f}", delta=round(cambio_r, 2))

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
                u_col1.metric("Salario (w*/p2)", f"{w_f:.2f}", delta=round(cambio_w_f, 2))
                u_col2.metric("Renta (r*/p2)", f"{r_f:.2f}", delta=round(cambio_r_f, 2))
                
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
        
        else:  # Autarquía
            u_col1, u_col2 = st.columns(2)
            u_col1.metric("Precio Nacional (p1/p2)", f"{p1_1_sol:.2f}")
            u_col2.metric("Precio Extranjero (p1/p2)", f"{p1_2_sol:.2f}")

            tab_aut_nac, tab_aut_ext = st.tabs(["🏠 Nacional", "🌍 Extranjero"])
            
            with tab_aut_nac:
                u_col1, u_col2 = st.columns(2)
                u_col1.metric("Salario (w/p2)", f"{w_1:.2f}")
                u_col2.metric("Renta (r/p2)", f"{r_1:.2f}")

                st.dataframe(pd.DataFrame({
                    "Bien": ["1", "2"],
                    "Producción": [f"{Y1_1:.1f}", f"{Y2_1:.1f}"],
                    "Consumo": [f"{C1_1:.1f}", f"{C2_1:.1f}"]
                }), hide_index=True)

                st.write("Asignación de los factores")
                st.dataframe(pd.DataFrame({
                    "Bien": ["1", "2"],
                    "Trabajo": [f"{L1_1_sol:.1f}", f"{L2_1_sol:.1f}"],
                    "Capital": [f"{K1_1_sol:.1f}", f"{K2_1_sol:.1f}"]
                }), hide_index=True)
            
            with tab_aut_ext:
                u_col1, u_col2 = st.columns(2)
                u_col1.metric("Salario (w*/p2)", f"{w_2:.2f}")
                u_col2.metric("Renta (r*/p2)", f"{r_2:.2f}")

                st.write("Producción y Consumo")
                st.dataframe(pd.DataFrame({
                    "Bien": ["1", "2"],
                    "Producción": [f"{Y1_2:.1f}", f"{Y2_2:.1f}"],
                    "Consumo": [f"{C1_2:.1f}", f"{C2_2:.1f}"]
                }), hide_index=True)

                st.write("Asignación de los factores")
                st.dataframe(pd.DataFrame({
                    "Bien": ["1", "2"],
                    "Trabajo": [f"{L1_2_sol:.1f}", f"{L2_2_sol:.1f}"],
                    "Capital": [f"{K1_2_sol:.1f}", f"{K2_2_sol:.1f}"]
                }), hide_index=True)
    
    

except Exception as e:
    st.warning(f"⚠️ El modelo no pudo converger con estos parámetros. Esto suele ocurrir cuando las dotaciones son muy extremas o las intensidades de factores son idénticas: {e}")