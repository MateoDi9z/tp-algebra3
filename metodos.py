"""
Matriz = [[1,2,0,0], [-1,3,1,0], [0,2,2,1],[0,0,1,1]]
vector = [5,8,14,7]

Matriz2 = [[1,-1,0,0,0], [2,5,-3,0,0], [0,4,2,1,0],[0,0,1,3,-4],[0,0,0,2,2]]
vector2 = [-1.5,43.5,9,-4.4,3.2]
"""
import numpy as np
from numpy.ma.core import indices


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

    print(a)
    print(b)
    return b


iii = 0
def solve_gauss_pivoteo(a, b):
    n = len(b)
    v=0
    global iii
    iii += 1

    print("LLAME A LA FUNCION", iii)

    permutacion = []
    for z in range(n):
        permutacion.append(z)

    for j in range(n):
        max = a[j, j]
        indice = j

        for i in range(j+1, n):
            if a[i, j] > max:
                max = a[i, j]
                indice = i

        t = permutacion[v]
        permutacion[v] = permutacion[indice]
        permutacion[indice] = t

        a = a[permutacion, :]
        x = b[v]
        b[v] = b[indice]
        b[indice] = x

        v += 1
        divisor = a[j, j]
        if divisor == 0: continue
        for k in range(0, n):
            a[j, k] = a[j, k] / divisor
        b[j] = b[j] / divisor

        for l in range(0, n-1):
            if j+l > n-1: continue
            d = 0

            for m in range(0, n):
                d = a[j+l, j]
                a[j+l, m] = a[j+l, m] - a[j, m] * d

            b[j+l] = b[j+l] - b[j] * d

    print(a)
    print(b)
    return b


"""
    A = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    n = len(b)
    v=0

    for j in range(n):
        max = a[j, j]
        indice = 0

        for i in range(j+1, n):
            if a[i, j] > max:
                max = a[i, j]
                indice = i
        t = a[v]
        x = b[v]
        a[v] = a[indice]
        b[v] = b[indice]
        a[indice] = t
        b[indice] = x
        v += 1

        divisor = a[j, j]
        if divisor == 0: continue
        for k in range(0, n):
            a[j, k] = a[j, k] / divisor
        b[j] = b[j] / divisor

        for l in range(0, n-1):
            if j+l > n-1: continue
            d = 0

            for m in range(0, n):
                d = a[j+l, j]
                a[j+l, m] = a[j+l, m] - a[j, m] * d

            b[j+l] = b[j+l] - b[j] * d

    print(a, b)
    return b
"""