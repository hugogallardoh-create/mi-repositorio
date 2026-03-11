import streamlit as st
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import root_scalar
from scipy.optimize import minimize_scalar

#################################
### FUNCIONES DE PRODUCCION #####
#################################

class cobb_douglas:
        def __init__(self, A1, A2, alpha1, alpha2):
            self.A1 = A1
            self.A2 = A2
            self.alpha1 = alpha1
            self.alpha2 = alpha2

        def requerimiento_factor_1l(self, w1, w2):
            w1 = max(w1, 1e-12)
            w2 = max(w2, 1e-12)
            return (self.alpha1/(1-self.alpha1))**(1-self.alpha1) * w1**(self.alpha1-1) * w2**(1-self.alpha1) / self.A1
        
        def requerimiento_factor_1k(self, w1, w2):
            w1 = max(w1, 1e-12)
            w2 = max(w2, 1e-12)
            return ((1-self.alpha1)/(self.alpha1))**(self.alpha1) * w2**(-self.alpha1) * w1**(self.alpha1) / self.A1
                    
        def requerimiento_factor_2l(self, w1, w2):
            w1 = max(w1, 1e-12)
            w2 = max(w2, 1e-12)
            return (self.alpha2/(1-self.alpha2))**(1-self.alpha2) * w1**(self.alpha2-1) * w2**(1-self.alpha2) / self.A2
        
        def requerimiento_factor_2k(self, w1, w2):
            w1 = max(w1, 1e-12)
            w2 = max(w2, 1e-12)
            return ((1-self.alpha2)/(self.alpha2))**(self.alpha2) * w2**(-self.alpha2) * w1**(self.alpha2) / self.A2

        def get_ppf(self, L_val, K_val):
            # La lógica de asignación eficiente (l1_range y k1_range) 
            # se mantiene igual porque la PFT no afecta la relación de marginalidades
            c1, c2 = self.alpha1/(1-self.alpha1), self.alpha2/(1-self.alpha2)
            l1_range = np.linspace(0.1, L_val-0.1, 100)
            
            # Esta fórmula determina la Curva de Contrato en la Caja de Edgeworth
            k1_range = (c2 * K_val * l1_range) / (c1 * L_val + (c2 - c1) * l1_range)
            
            # Aplicamos la PFT (A1 y A2) al calcular el producto final
            q1 = self.A1 * (l1_range**self.alpha1) * (k1_range**(1-self.alpha1))
            q2 = self.A2 * ((L_val-l1_range)**self.alpha2) * ((K_val-k1_range)**(1-self.alpha2))
            
            return q1, q2
        



class ces_production:
        def __init__(self, A1, A2, d1l, d1k, d2l, d2k, sigma1, sigma2):
            self.A1, self.A2 = A1, A2
            self.d1l, self.d1k = d1l, d1k
            self.d2l, self.d2k = d2l, d2k
            self.s1, self.s2 = sigma1, sigma2
            
            # Parámetros rho derivados de sigma
            self.rho1 = (sigma1 - 1) / sigma1 
            self.rho2 = (sigma2 - 1) / sigma2

        def _unit_cost(self, w, r, sector=1):
            """Calcula el costo unitario (dual de la CES)"""
            if sector == 1:
                a, b, s, A = self.d1l, self.d1k, self.s1, self.A1
            else:
                a, b, s, A = self.d2l, self.d2k, self.s2, self.A2
            
            # Fórmula del costo unitario para CES
            return (1/A) * (a**s * w**(1-s) + b**s * r**(1-s))**(1/(1-s))

        def requerimiento_factor_1l(self, w, r):
            # Derivada del costo unitario respecto a w (Lema de Shephard)
            w = max(w, 1e-12)
            r = max(r, 1e-12)
            term_parentesis = (self.d1l**self.s1 * w**(1-self.s1) + self.d1k**self.s1 * r**(1-self.s1))
            return (1/self.A1) * (term_parentesis**(self.s1/(1-self.s1))) * (self.d1l**self.s1 * w**(-self.s1))

        def requerimiento_factor_1k(self, w, r):
            w = max(w, 1e-12)
            r = max(r, 1e-12)
            term_parentesis = (self.d1l**self.s1 * w**(1-self.s1) + self.d1k**self.s1 * r**(1-self.s1))
            return (1/self.A1) * (term_parentesis**(self.s1/(1-self.s1))) * (self.d1k**self.s1 * r**(-self.s1))

        def requerimiento_factor_2l(self, w, r):
            w = max(w, 1e-12)
            r = max(r, 1e-12)
            term_parentesis = (self.d2l**self.s2 * w**(1-self.s2) + self.d2k**self.s2 * r**(1-self.s2))
            return (1/self.A2) * (term_parentesis**(self.s2/(1-self.s2))) * (self.d2l**self.s2 * w**(-self.s2))

        def requerimiento_factor_2k(self, w, r):
            w = max(w, 1e-12)
            r = max(r, 1e-12)
            term_parentesis = (self.d2l**self.s2 * w**(1-self.s2) + self.d2k**self.s2 * r**(1-self.s2))
            return (1/self.A2) * (term_parentesis**(self.s2/(1-self.s2))) * (self.d2k**self.s2 * r**(-self.s2))

        def get_ppf(self, L_total, K_total):
            l1_range = np.linspace(0.01, L_total - 0.01, 50)
            k1_efficient = []

            for l1 in l1_range:
                # En lugar de buscar el 0, minimizamos la diferencia al cuadrado
                def objective(k1):
                    rmts1 = (self.d1l / self.d1k) * (l1 / k1)**(-1/self.s1)
                    rmts2 = (self.d2l / self.d2k) * ((L_total - l1) / (K_total - k1))**(-1/self.s2)
                    return (rmts1 - rmts2)**2
                
                # minimize_scalar respeta los límites (bounds) sin romper las derivadas
                res = minimize_scalar(
                    objective, 
                    bounds=(1e-5, K_total - 1e-5), 
                    method='bounded'
                )
                k1_efficient.append(res.x)
            
            k1_efficient = np.array(k1_efficient)
            
            # Calcular producción con tecnología CES
            q1 = self.A1 * (self.d1l * l1_range**self.rho1 + self.d1k * k1_efficient**self.rho1)**(1/self.rho1)
            q2 = self.A2 * (self.d2l * (L_total - l1_range)**self.rho2 + self.d2k * (K_total - k1_efficient)**self.rho2)**(1/self.rho2)
            
            return q1, q2
        

#################################
###  FUNCIONES DE CONSUMO   #####
#################################

    
class cobb_douglas_utilidad:
        def __init__(self, alpha):
            self.alpha = alpha

        def demanda1(self, p, I):
            C = self.alpha * I / p
            return C

        def demanda2(self, p, I):
            C = (1-self.alpha) * I / p
            return C
        
class utilidad_lineal:
    def __init__(self, alpha, beta):
        """
        U = alpha * x + beta * y
        """
        self.alpha = alpha
        self.beta = beta

    def demanda1(self, p1, I):
        # Comparamos utilidades marginales ponderadas por el precio
        if (self.alpha / p1) > (self.beta):
            return I / p1
        elif (self.alpha / p1) == (self.beta):
            # Indiferencia: cualquier punto en la restricción es óptimo
            # Se suele devolver el punto medio o un rango
            return I / (2 * p1) 
        else:
            return 0

    def demanda2(self, p1, I):
        if (self.beta ) > (self.alpha / p1):
            return I 
        elif (self.beta ) == (self.alpha / p1):
            return I / (2)
        else:
            return 0

class utilidad_leontief:
    def __init__(self, a, b):
        """
        Utilidad = min(a*x1, b*x2)
        a: coeficiente del bien 1
        b: coeficiente del bien 2
        """
        self.a = a
        self.b = b

    def demanda1(self, p1, I):
        # x1 = (b * I) / (b*p1 + a*p2)
        denominador = (self.b * p1) + (self.a)
        return (self.b * I) / denominador

    def demanda2(self, p1, I):
        # x2 = (a * I) / (b*p1 + a*p2)
        denominador = (self.b * p1) + (self.a)
        return (self.a * I) / denominador
    

class utilidad_ces:
    def __init__(self, delta1, delta2, sigma):
        self.d1 = delta1
        self.d2 = delta2
        self.s = sigma

    def demanda1(self, p1, I):
        # Numerador: I * (delta1^sigma) * (px^-sigma)
        numerador = I * (self.d1**self.s) * (p1**(-self.s))
        # Denominador: (delta1^sigma * px^(1-sigma)) + (delta2^sigma * py^(1-sigma))
        denominador = (self.d1**self.s * p1**(1-self.s)) + (self.d2**self.s * 1**(1-self.s))
        return numerador / denominador

    def demanda2(self, p1, I):
        numerador = I * (self.d2**self.s) * (1**(-self.s))
        denominador = (self.d1**self.s * p1**(1-self.s)) + (self.d2**self.s * 1**(1-self.s))
        return numerador / denominador


class utilidad_cuasilineal:
    def __init__(self, alpha):
        self.alpha = alpha

    def demanda1(self, px, I):
        # Si I < alpha, el consumidor gasta todo en x
        # Si I >= alpha, la demanda es constante en alpha/px
        if I < self.alpha:
            return I / px
        else:
            return self.alpha / px

    def demanda2(self, px, I):
        # El bien y es el resto del ingreso
        # y = I - px * (alpha/px) = I - alpha
        return max(0, I - self.alpha)
    

class utilidad_stonegery:
    def __init__(self, alpha, gamma1, gamma2):
        self.alpha = alpha
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        # p2 es implícitamente 1
        self.p2 = 1 

    def ingreso_supernumerario(self, p1, I):
        """
        Calcula el ingreso tras cubrir subsistencia.
        Como p2 = 1, el costo de subsistencia es (p1*gamma1 + gamma2)
        """
        I_super = I - (p1 * self.gamma1 + self.gamma2)
        return max(0, I_super)

    def demanda1(self, p1, I):
        I_s = self.ingreso_supernumerario(p1, I)
        # x1 = gamma1 + (alpha * I_super) / p1
        return self.gamma1 + (self.alpha * I_s) / p1

    def demanda2(self, p1, I):
        I_s = self.ingreso_supernumerario(p1, I)
        # x2 = gamma2 + (1 - alpha) * I_super (ya que p2 = 1)
        return self.gamma2 + (1 - self.alpha) * I_s