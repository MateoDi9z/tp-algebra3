"""
Matriz = [[1,2,0,0], [-1,3,1,0], [0,2,2,1],[0,0,1,1]]
vector = [5,8,14,7]

Matriz2 = [[1,-1,0,0,0], [2,5,-3,0,0], [0,4,2,1,0],[0,0,1,3,-4],[0,0,0,2,2]]
vector2 = [-1.5,43.5,9,-4.4,3.2]
"""
import numpy as np
from numpy.ma.core import indices

import numpy as np
from scipy.sparse import identity, csc_matrix, spmatrix, lil_matrix


def permutar_filas(A: spmatrix, i: int, j: int) -> spmatrix:
    """
    Devuelve una copia de la matriz A con las filas i y j permutadas.

    Parámetros:
        A: Matriz dispersa (scipy.sparse)
        i: Índice de la primera fila a permutar
        j: Índice de la segunda fila a permutar

    Retorna:
        Nueva matriz con las filas permutadas.
    """
    n_rows = A.shape[0]
    P = identity(n_rows, format='csc').toarray()
    P[[i, j]] = P[[j, i]]
    P_perm = csc_matrix(P)
    return P_perm @ A

def permutar_columnas(A: spmatrix, i: int, j: int) -> spmatrix:
    """
    Devuelve una copia de la matriz A con las columnas i y j permutadas.

    Parámetros:
        A: Matriz dispersa (scipy.sparse)
        i: Índice de la primera columna a permutar
        j: Índice de la segunda columna a permutar

    Retorna:
        Nueva matriz con las columnas permutadas.
    """
    n_cols = A.shape[1]
    P = identity(n_cols, format='csc').toarray()
    P[:, [i, j]] = P[:, [j, i]]
    P_perm = csc_matrix(P)
    return A @ P_perm

def solve_tridiagonal_system(A, b):
    n = len(A) - 1

    for i in range(0,n+1):
        # CASO BASE (n=0)
        if i == 0:
            A[0][1] = A[0][1] / A[0][0] # fila / base
            b[0] = b[0]/A[0][0]         # b[0] / base
            A[0][0] = 1                 # base = 1
            v = A[1][0]                 # guardar el de abajo

            for j in range(0,2):
                A[1][j] = A[1][j] - v*A[0][j]

            b[1] = b[1] - b[0] * v

        elif i == n:
            b[n] = b[n] / A[n][n]
            A[n][n] = 1

        else:
            divisor = A[i][i]
            b[i] = b[i] / divisor

            for j in range(i,i+2):
                A[i][j] = A[i][j] / divisor

            v = A[i+1][i]
            b[i+1]= b[i+1] - b[i]*v

            for j in range(i,i+2):
                A[i+1][j] = A[i+1][j] - v*A[i][j]

    # Triangulación Ascendente
    for i in range(n,0,-1):
        v = A[i-1][i]
        A[i-1][i] = 0 # v - v
        b[i-1] = b[i-1] - v * b[i]

    return b

def solve_gauss_pivoteo(A, b):
    # TODO: all.
    return

def solve_tridiagonal_system_megaopti(a, b):
    n = len(b)

    diagonal = []
    diagonal_inf = []
    diagonal_sup = []

    for i in range(0,n):
        diagonal.append(a[i, i])
        if i == n-1: continue
        diagonal_sup.append(a[i, i+1])
        diagonal_inf.append(a[i+1, i])

    for i in range(0, n):
        if i == n - 1:
            base = diagonal[i]
            #diagonal[i] = 1
            b[i] /= base
            continue

        base = diagonal[i]
        #diagonal[i] = 1
        diagonal_sup[i] /= base
        b[i] /= base
        multiplicador = diagonal_inf[i]
        diagonal_inf[i] = 0
        diagonal[i + 1] -= multiplicador * diagonal_sup[i]
        b[i + 1] -= multiplicador * b[i]

    # Jordan
    for j in range(n-1, 0, -1):
        multiplicador = diagonal_sup[j-1]
        #diagonal_sup[j-1] = 0
        b[j-1] -= multiplicador * b[j]

    return b

def solve_tridiagonal_system_optimizado(a, b):
    n = len(b)
    diagonal = []
    diagonal_inf = []
    diagonal_sup = []

    for i in range(0,n):
        diagonal.append(a[i, i])
        if i == n-1: continue
        diagonal_sup.append(a[i, i+1])
        diagonal_inf.append(a[i+1, i])

    b[0] /= diagonal[0]

    for i in range(1, n-1):
        b[i] -= diagonal_inf[i-1] * b[i-1]
        diagonal[i] -= diagonal_inf[i-1] * diagonal_sup[i-1]
        diagonal_sup[i] /= diagonal[i]
        b[i] /= diagonal[i]

    b[n-1] /= diagonal[n-1]

    # Jordan
    for j in range(n-1, 0, -1):
        b[j-1] -= diagonal_sup[j-1] * b[j]

    return b


iii = 0
def solve_gauss_pivoteo(A, b):
    n = len(b)
    b = lil_matrix(b)  # Matrix editable
    A = lil_matrix(A)  # Matrix editable

    for j in range(n):
        max_val = abs(A[j, j])
        indice = j
        for i in range(j+1 , n):
             if abs(A[i, j]) > max_val :
                max_val = abs(A[i, j])
                indice = i

        A = lil_matrix(permutar_filas(A, indice, j))
        b = lil_matrix(permutar_filas(b, indice, j))

        divisor = A[j, j]
        for k in range(j,n):
            A[j, k] = A[j, k] / divisor
        b[j] = b[j] / divisor
        for l in range(j+1, n) :
            d = A[l, j]
            b[l] -= b[j] * d
            for m in range(j,n):
                A[l, m] -= A[j, m] * d

    for i in range(n-1,0,-1):
        for j in range(i,0,-1):
            A[j-1, i] -= A[i, i] * A[j-1, i]
            b[j-1] -= b[i] * A[j-1, i]

    return csc_matrix(b)
