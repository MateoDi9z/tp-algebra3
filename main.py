# TP Simulación Física en Videojuegos: Difusión de Calor 2D

import numpy as np
import time

from pyparsing import results
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import spsolve

from metodos import *
from funciones import graficar_listas, graficar_errores


# Construcción matriz A para método implícito
def construir_matriz(nx, ny, dx, dy, dt, alpha):
    Nix, Niy = nx - 2, ny - 2
    Ix = identity(Nix)
    Iy = identity(Niy)

    main_diag_x = 2 * (1/dx**2 + 1/dy**2) * np.ones(Nix)
    off_diag_x = -1/dx**2 * np.ones(Nix - 1)
    Tx = diags([off_diag_x, main_diag_x, off_diag_x], [-1, 0, 1])

    off_diag_y = -1/dy**2 * np.ones(Niy - 1)
    Ty = diags([off_diag_y, off_diag_y], [-1, 1], shape=(Niy, Niy))

    L = kron(Iy, Tx) + kron(Ty, Ix)
    A = identity(Nix*Niy) - dt * alpha * L
    return A

# Inicializar temperatura y condiciones de frontera
def inicializar_T(nx, ny):
    T = np.ones((ny, nx)) * 25
    T[:, 0] = 100    # borde izquierdo
    T[:, -1] = 50    # borde derecho
    T[0, :] = 0      # borde superior
    T[-1, :] = 75    # borde inferior
    # Fuente interna caliente
    T[ny//2 - 1:ny//2 + 2, nx//2 - 1:nx//2 + 2] = 200
    return T

# TODO: Métodos a desarrollar
def optimizado(A, b):
    """Resolución de un sistema de ecuaciones lineales optimizado
        cuando la matriz A es tridiagonal.
    """
    return solve_tridiagonal_system_optimizado(A, b)

def gauss_pivoteo(A, b):
    """Método de Gauss con pivoteo parcial"""
    return solve_gauss_pivoteo(A, b)

# Un paso de simulación con el método implícito y método de solución
def paso_simulacion(T, A, nx, ny, dx, dy, dt, alpha, metodo_solucion: str):
    b = T[1:-1, 1:-1].copy()
    # Incorporar condiciones de borde en b
    b[:, 0] += dt * alpha * T[1:-1, 0] / dx**2
    b[:, -1] += dt * alpha * T[1:-1, -1] / dx**2
    b[0, :] += dt * alpha * T[0, 1:-1] / dy**2
    b[-1, :] += dt * alpha * T[-1, 1:-1] / dy**2
    b = b.flatten()

    if metodo_solucion == 'directo':
        T_vec = spsolve(A, b)
    elif metodo_solucion == 'optimizado':
        T_vec = optimizado(A, b)
    elif metodo_solucion == 'gauss_pivoteo':
        T_vec  = gauss_pivoteo(A, b)
    else:
        raise ValueError("Método de solución no reconocido")

    T_new = T.copy()
    T_new[1:-1, 1:-1] = T_vec.reshape((ny - 2, nx - 2))
    # Mantener la fuente de calor interna fija
    T_new[ny//2 - 1:ny//2 + 2, nx//2 - 1:nx//2 + 2] = 200
    return T_new

# Simular múltiples pasos, medir tiempos
def simular(nx, ny, dt, alpha, pasos, metodo_solucion):
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    A = construir_matriz(nx, ny, dx, dy, dt, alpha)
    T = inicializar_T(nx, ny)

    tiempos = []
    for _ in range(pasos):
        start = time.time()
        T = paso_simulacion(T, A, nx, ny, dx, dy, dt, alpha, metodo_solucion)
        end = time.time()
        tiempos.append(end - start)

    tiempo_promedio = np.mean(tiempos)
    return T, tiempo_promedio

# Error RMS relativo
def error_rms(T_ref, T):
    return np.sqrt(np.mean((T_ref - T)**2)) / np.sqrt(np.mean(T_ref**2))

# --------- Experimentos ---------

resoluciones = [10, 20, 30, 50, 70]
dt = 0.1
alpha = 0.01
pasos = 10
metodos = ['directo', 'optimizado', 'gauss_pivoteo']

# TODO: correr experimientos, obtener resultados y compararlos
metodo_seleccionado = 2

tiempos = []
errors = []

print("|\tn\t|\ttiempo\t|\tTiempo de Referencia\t|\tError\t|")
resultado, t1 = simular(10, 10, dt, alpha, pasos, metodos[metodo_seleccionado])

"""
for res in resoluciones[:1]:
    resultado, t1 = simular(res, res, dt, alpha, pasos, metodos[metodo_seleccionado])
    tiempos.append(t1*1000)
    ref, t2 = simular(res, res, dt, alpha, pasos, metodos[0])
    error = error_rms(ref, resultado)
    errors.append(error)

    print(f"|\t{res}\t|\t{round(t1*1000, 1)}ms\t|\t\t{round(t2*1000, 9)}ms\t\t|\t{round(error, 2)} \t|")
"""
"""
graficar_listas(
    resoluciones,
    tiempos,
    f'Metodo {metodos[metodo_seleccionado]}'
)"""

"""
graficar_errores(
    resoluciones,
    errors,
    f'Metodo {metodos[metodo_seleccionado]}'
)"""