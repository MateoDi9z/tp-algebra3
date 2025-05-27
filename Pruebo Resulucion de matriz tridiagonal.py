Matriz = [[1,2,0,0], [-1,3,1,0], [0,2,2,1],[0,0,1,1]]
vector = [5,8,14,7]

Matriz2 = [[1,-1,0,0,0], [2,5,-3,0,0], [0,4,2,1,0],[0,0,1,3,-4],[0,0,0,2,2]]
vector2 = [-1.5,43.5,9,-4.4,3.2]


def solveTridiagonalSistem(A, b):
    n = len(A)-1
    for i in range(0,n+1):
        if i == 0:
            A[0][1] = A[0][1]/A[0][0]
            b[0] = b[0]/A[0][0]
            A[0][0] = 1
            v = A[1][0]
            for j in range(0,2):
                A[1][j] = A[1][j] - v*A[0][j]
            b[1] = b[1] - b[0] * v
        elif i == n:
            b[n] = b[n]/A[n][n]
            A[n][n] = 1
        else:
            divisor = A[i][i]
            b[i]=b[i]/divisor
            for j in range(i,i+2):
                A[i][j] = A[i][j] / divisor

            v = A[i+1][i]
            b[i+1]= b[i+1] - b[i]*v
            for j in range(i,i+2):
                A[i+1][j] = A[i+1][j] - v*A[i][j]
    for i in range(n,0,-1):
        v = A[i-1][i]
        A[i-1][i] = v - v
        b[i-1] = b[i-1] - v * b[i]
    return b

print(solveTridiagonalSistem(Matriz2,vector2))



