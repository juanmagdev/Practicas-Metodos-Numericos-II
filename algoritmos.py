# -*- coding: utf-8 -*-

from numpy import *
from numpy.linalg import *
from numpy import abs, sum, max, min


def conjugada(A):
    return conjugate(transpose(A))


def traza(A):
    return sum(diag(A))

# función norma_vec(), norma de un vector, si p = "inf" norma infinito, si no, norma ...**1/p


def norma_vec(X, p):
    XX = array(X, dtype=complex)
    normainf = max(abs(XX))
    if p == inf:
        return normainf
    elif p >= 1:
        if normainf <= 1e-15:
            return sum(abs(XX)**p)**(1/p)
        else:
            return normainf*sum((abs(XX)/normainf)**p)**(1/p)
    else:
        return "Error norma_vec: valor de p."


# función conv_norma_vec() , EJERCICIO 10 Practica 2
def conv_norma_vec(X):
    print("\n Vector: X = ", X)
    normainf = norma_vec(X, inf)
    print("Norma infinito = ", normainf)
    error = 1.
    p = 0
    while error >= 1e-15 and p < 200:
        p = p+1
        normap = norma_vec(X, p)
        if normainf <= 1e-15:
            error = abs(normap - normainf)
            print("Norma ", p, " = ", normap, " Error absoluto = ", error)
        else:
            error = abs((normap - normainf)/normainf)
            print("Norma ", p, " = ", normap, " Error relativo = ", error)
    if error < 1e-15:
        print("Convergencia numérica alcanzada.")
    else:
        print("Número máximo de iteraciones alcanzado.")


# norma matricial
def norma_mat(A, p):
    A = array(A, dtype=complex)
    if p == inf:
        return max(sum(abs(A), axis=1))
    elif p == 1:
        return max(sum(abs(A), axis=0))
    elif p == 2:
        return sqrt(max(abs(eig(conjugada(A)@A)[0])))
    elif p == 'esp':
        return max(svd(A)[1])
    elif p == 'fro':
        return sqrt(sum(abs(A)**2))
    else:
        return "Error norma_mat: valor de p."

# Ejercicio 5 pracitca 3

def aprox_norma_mat(A, p, nva):
    m, n = shape(A)
    norma = 0.
    for i in range(nva):
        X = random.rand(n, 1)
        aux = norma_vec(A@X, p) / norma_vec(X, p)
        norma = max([norma, aux])
    print("Norma matricial: ", norma_mat(A, p))
    print("Aproximación: ", norma)

# Definición de la función descenso()
# Devuelve true si se ha conseguido la solucion, junto a la solución. False si no se ha conseguido la solución.


def descenso(A, B):
    m, n = shape(A)
    p, q = shape(B)
    if m != n or n != p or q < 1:
        return False, "Error descenso: error en las dimensiones."
    if min(abs(diag(A))) < 1e-200:
        return False, "Error descenso: matriz singular."
    if A.dtype == complex or B.dtype == complex:
        X = zeros((n, q), dtype=complex)
    else:
        X = zeros((n, q), dtype=float)
    for i in range(n):
        X[i, :] = B[i, :]
        if i != 0:
            X[i, :] -= A[i, :i]@X[:i, :]
        X[i, :] = X[i, :]/A[i, i]
    return True, X


# Definición de la función descenso1()
def descenso1(A, B):
    (m, n) = shape(A)
    (p, q) = shape(B)
    if m != n or n != p or q < 1:
        return False, "descenso: error en las dimensiones"
    # if min(abs(diag(A))) < 1e-100:
    #    return False, "descenso: matriz singular"
    if A.dtype == complex or B.dtype == complex:
        X = zeros((n, q), dtype=complex)
    else:
        X = zeros((n, q), dtype=float)
    for i in range(n):
        X[i, :] = B[i, :]
        if i != 0:
            X[i, :] = X[i, :] - A[i, :i]@X[:i, :]
        # X[i, :] = X[i, :]/A[i, i]
    return True, X


# Definición de la función remonte()
def remonte(A, B):
    m, n = shape(A)
    p, q = shape(B)
    if m != n or n != p or q < 1:
        return False, "Error remonte: error en las dimensiones."
    if min(abs(diag(A))) < 1e-200:
        return False, "Error remonte: matriz singular."
    if A.dtype == complex or B.dtype == complex:
        X = zeros((n, q), dtype=complex)
    else:
        X = zeros((n, q), dtype=float)
    for i in range(n):
        for k in range(q):
            j = n-1-i
            X[j, k] = B[j, k]
            if i != k:
                X[j, k] -= A[j, (j+1):n]@X[(j+1):n, k]
            X[j, k] = X[j, k]/A[j, j]
    return True, X

# Sólo se usa en el caso en que la diagonal sea todo 1


def remonte1(A, B):
    m, n = shape(A)
    p, q = shape(B)
    if m != n or n != p or q < 1:
        return False, "Error descenso: error en las dimensiones."
    if min(abs(diag(A))) < 1e-200:
        return False, "Error descenso: matriz singular."
    if A.dtype == complex or B.dtype == complex:
        X = zeros((n, q), dtype=complex)
    else:
        X = zeros((n, q), dtype=float)
    # Desde n-1 hasta 0 con salto -1  (ponemos -1 porque el -1 no está incluido, para en 0)
    for i in range(n-1, -1, -1):
        X[i, :] = B[i, :]
        if i != n-1:
            X[i, :] -= A[i, i+1:]@X[i+1:, :]
    return True, X


# Función descenso_1diag()
def descenso_1diag(A, B):
    (m, n) = shape(A)
    (p, q) = shape(B)
    if m != n or n != p or q < 1:
        return False, "descenso: error en las dimensiones"
    if min(abs(diag(A))) < 1e-100:
        return False, "descenso: matriz singular"
    if A.dtype == complex or B.dtype == complex:
        X = zeros((n, q), dtype=complex)
    else:
        X = zeros((n, q), dtype=float)
    for i in range(n):
        X[i, :] = B[i, :]
        if i != 0:
            X[i, i] = X[i, i] - A[i, i-1]*X[i-1, :]
        X[i, :] = X[i, :]/A[i, i]
    return True, X

# Función remonte_1diag()


def remonte_1diag(A, B):
    (m, n) = shape(A)
    (p, q) = shape(B)
    if m != n or n != p or q < 1:
        return False, "remonte: error en las dimensiones"
    if min(abs(diag(A))) < 1e-100:
        return False, "remonte: matriz singular"
    if A.dtype == complex or B.dtype == complex:
        X = zeros((n, q), dtype=complex)
    else:
        X = zeros((n, q), dtype=float)
    for i in range(n-1, -1, -1):
        X[i, :] = B[i, :]
        if i != n-1:
            X[i, :] = X[i, :] - A[i, i+1:]@X[i+1:, :]
        X[i, :] = X[i, :]/A[i, i]
    return True, X


def gauss_pp(A, B):
    m, n = shape(A)
    p, q = shape(B)
    if m != n or n != p or q < 1:
        return False, "Error gauss_pp: error en las dimensiones."
    if A.dtype == complex or B.dtype == complex:
        gaussA = array(A, dtype=complex)
        gaussB = array(B, dtype=complex)
    else:
        gaussA = array(A, dtype=float)
        gaussB = array(B, dtype=float)
    for k in range(n-1):
        pos = argmax(abs(gaussA[k:, k]))
        ik = pos+k
        if ik != k:
            gaussA[[ik, k], :] = gaussA[[k, ik], :]
            gaussB[[ik, k], :] = gaussB[[k, ik], :]
        if abs(gaussA[k, k]) >= 1e-200:
            for i in range(k+1, n):
                gaussA[i, k] = gaussA[i, k]/gaussA[k, k]
                gaussA[i, k+1:] -= gaussA[i, k]*gaussA[k, k+1:]
                gaussB[i, :] -= gaussA[i, k]*gaussB[k, :]
    exito, X = remonte(gaussA, gaussB)
    return exito, X


#Si nos piden las matrices gaussA, gaussB (MA = triu(gaussA), MB = gaussB)
def gauss_pp2(A, B):
    m, n = shape(A)
    p, q = shape(B)
    if m != n or n != p or q < 1:
        return False, "Error gauss_pp: error en las dimensiones."
    if A.dtype == complex or B.dtype == complex:
        gaussA = array(A, dtype=complex)
        gaussB = array(B, dtype=complex)
    else:
        gaussA = array(A, dtype=float)
        gaussB = array(B, dtype=float)
    for k in range(n-1):
        pos = argmax(abs(gaussA[k:, k]))
        ik = pos+k
        if ik != k:
            gaussA[[ik, k], :] = gaussA[[k, ik], :]
            gaussB[[ik, k], :] = gaussB[[k, ik], :]
        if abs(gaussA[k, k]) >= 1e-200:
            for i in range(k+1, n):
                gaussA[i, k] = gaussA[i, k]/gaussA[k, k]
                gaussA[i, k+1:] -= gaussA[i, k]*gaussA[k, k+1:]
                gaussB[i, :] -= gaussA[i, k]*gaussB[k, :]
    exito, X = remonte(gaussA, gaussB)
    return gaussA, gaussB


def gaussjordan_pp(A, B):
    (m, n) = shape(A)
    (p, q) = shape(B)
    if m != n or n != p or q < 1:
        return False, "gaussjordan_pp: error en las dimensiones"
    if A.dtype == complex or B.dtype == complex:
        gjA = array(A, dtype=complex)
        gjB = array(B, dtype=complex)
    else:
        gjA = array(A, dtype=float)
        gjB = array(B, dtype=float)
    for k in range(n):
        pos = argmax(abs(gjA[k:n, k]))
        ik = pos+k
        if ik != k:
            gjA[[ik, k], :] = gjA[[k, ik], :]
            gjB[[ik, k], :] = gjB[[k, ik], :]
        if abs(gjA[k, k]) >= 1e-15:
            for i in range(k):
                gjA[i, k] = gjA[i, k]/gjA[k, k]
                gjA[i, k+1:] = gjA[i, k+1:]-gjA[i, k]*gjA[k, k+1:]
                gjB[i, :] = gjB[i, :]-gjA[i, k]*gjB[k, :]
            for i in range(k+1, n):
                gjA[i, k] = gjA[i, k]/gjA[k, k]
                gjA[i, k+1:] = gjA[i, k+1:]-gjA[i, k]*gjA[k, k+1:]
                gjB[i, :] = gjB[i, :]-gjA[i, k]*gjB[k, :]
    if min(abs(diag(gjA))) < 1e-15:
        return False, "gaussjordan_pp: matriz singular"
    if A.dtype == complex or B.dtype == complex:
        X = zeros((n, q), dtype=complex)
    else:
        X = zeros((n, q), dtype=float)
    for i in range(n):
        X[i, :] = gjB[i, :]/gjA[i, i]
    return True, X


#Si nos piden las matrices gJA, gJB (MA = triu(gJA) - triu(gJA, k=1), MB = gJB
# MA tabien se puede calcular como MA = diagflat(diag(gjA), gjB)
def gaussjordan_pp2(A, B):
    (m, n) = shape(A)
    (p, q) = shape(B)
    if m != n or n != p or q < 1:
        return False, "gaussjordan_pp2: error en las dimensiones"
    if A.dtype == complex or B.dtype == complex:
        gjA = array(A, dtype=complex)
        gjB = array(B, dtype=complex)
    else:
        gjA = array(A, dtype=float)
        gjB = array(B, dtype=float)
    for k in range(n):
        pos = argmax(abs(gjA[k:n, k]))
        ik = pos+k
        if ik != k:
            gjA[[ik, k], :] = gjA[[k, ik], :]
            gjB[[ik, k], :] = gjB[[k, ik], :]
        if abs(gjA[k, k]) >= 1e-15:
            for i in range(k):
                gjA[i, k] = gjA[i, k]/gjA[k, k]
                gjA[i, k+1:] = gjA[i, k+1:]-gjA[i, k]*gjA[k, k+1:]
                gjB[i, :] = gjB[i, :]-gjA[i, k]*gjB[k, :]
            for i in range(k+1, n):
                gjA[i, k] = gjA[i, k]/gjA[k, k]
                gjA[i, k+1:] = gjA[i, k+1:]-gjA[i, k]*gjA[k, k+1:]
                gjB[i, :] = gjB[i, :]-gjA[i, k]*gjB[k, :]
    if min(abs(diag(gjA))) < 1e-15:
        return False, "gaussjordan_pp2: matriz singular"
    if A.dtype == complex or B.dtype == complex:
        X = zeros((n, q), dtype=complex)
    else:
        X = zeros((n, q), dtype=float)
    for i in range(n):
        X[i, :] = gjB[i, :]/gjA[i, i]
    return True, gjA, gjB


def gauss_1p(A, B):
    (m, n) = shape(A)
    (p, q) = shape(B)
    if m != n or n != p or q < 1:
        return False, "gauss_1p: error en las dimensiones"
    if A.dtype == complex or B.dtype == complex:
        gaussA = array(A, dtype=complex)
        gaussB = array(B, dtype=complex)
    else:
        gaussA = array(A, dtype=float)
        gaussB = array(B, dtype=float)
    for k in range(n-1):
        for ik in range(k, n):
            if abs(gaussA[ik, k]) > 1e-15:
                break
        if ik != k:
            gaussA[[ik, k], :] = gaussA[[k, ik], :]
            gaussB[[ik, k], :] = gaussB[[k, ik], :]
        if abs(gaussA[k, k]) >= 1e-15:
            for i in range(k+1, n):
                gaussA[i, k] = gaussA[i, k]/gaussA[k, k]
                gaussA[i, k+1:] = gaussA[i, k+1:]-gaussA[i, k]*gaussA[k, k+1:]
                gaussB[i, :] = gaussB[i, :]-gaussA[i, k]*gaussB[k, :]
    exito, X = remonte(gaussA, gaussB)
    return exito, X


def gaussjordan_1p(A, B):
    (m, n) = shape(A)
    (p, q) = shape(B)
    if m != n or n != p or q < 1:
        return False, "gaussjordan_1p: error en las dimensiones"
    if A.dtype == complex or B.dtype == complex:
        gjA = array(A, dtype=complex)
        gjB = array(B, dtype=complex)
    else:
        gjA = array(A, dtype=float)
        gjB = array(B, dtype=float)
    for k in range(n):
        for ik in range(k, n):
            if abs(gjA[ik, k]) > 1e-15:
                break
        if ik != k:
            gjA[[ik, k], :] = gjA[[k, ik], :]
            gjB[[ik, k], :] = gjB[[k, ik], :]
        if abs(gjA[k, k]) >= 1e-15:
            for i in range(k):
                gjA[i, k] = gjA[i, k]/gjA[k, k]
                gjA[i, k+1:] = gjA[i, k+1:]-gjA[i, k]*gjA[k, k+1:]
                gjB[i, :] = gjB[i, :]-gjA[i, k]*gjB[k, :]
            for i in range(k+1, n):
                gjA[i, k] = gjA[i, k]/gjA[k, k]
                gjA[i, k+1:] = gjA[i, k+1:]-gjA[i, k]*gjA[k, k+1:]
                gjB[i, :] = gjB[i, :]-gjA[i, k]*gjB[k, :]
    if min(abs(diag(gjA))) < 1e-15:
        return False, "gaussjordan_1p: matriz singular"
    if A.dtype == complex or B.dtype == complex:
        X = zeros((n, q), dtype=complex)
    else:
        X = zeros((n, q), dtype=float)
    for i in range(n):
        X[i, :] = gjB[i, :]/gjA[i, i]
    return True, X


# devuelve en una misma matriz la matriz L y la matriz U
# La matriz L es la matriz triangular inferior con unos en la diagonal, U triangular superior

# si queremos descomponer L y U
# e, LU = facto_lu(A)
# L = tril(LU, k = -1) + eye(n),        n dimensiones
# U = triu(LU)

def facto_lu(A):
    (m, n) = shape(A)
    if m != n:
        return False, "facto_lu: error en las dimensiones"
    if A.dtype == complex:
        lu = array(A, dtype=complex)
    else:
        lu = array(A, dtype=float)
    for k in range(n-1):
        if abs(lu[k, k]) < 1e-15:
            return False, "facto_lu: no existe la factorización"
        else:
            for i in range(k+1, n):
                lu[i, k] = lu[i, k]/lu[k, k]
                lu[i, k+1:] = lu[i, k+1:]-lu[i, k]*lu[k, k+1:]
    return True, lu


def metodo_lu(A, B):
    exito, lu = facto_lu(A)
    if exito:
        exito2, Y = descenso1(lu, B)
        exito3, X = remonte(lu, Y)
        if exito2 and exito3:
            return True, X
        else:
            return False, "metodo_lu: error en la resolución."
    else:
        return False, "metodo_lu: error en la factorización."


def admite_cholesky(A):
    m, n = shape(A)
    if m != n:
        return False, "NO ADMITE CHOLESKY: no es matriz cuadrada"
    if max(abs(A-transpose(A))) > 1e-10 :
        return False, "NO ADMITE CHOLESKY: no es simétrica"      
    determinante = A[0,0]
    i = 2
    while i<=n and determinante > 0:
        M = A[0:i,0:i]
        determinante = det(M)
        i += 1
    if i == n+1:
           return True, "ADMITE CHOLESKY"
    else:
           return False, "NO ADMITE CHOLESKY: no es definida positiva"


#para obtener la matriz hacer tril(chol)
def facto_cholesky(A):
    (m, n) = shape(A)
    if m != n:
        return False, "facto_cholesky: error en las dimensiones"
    if A.dtype == complex:
        chol = array(A, dtype=complex)
    else:
        chol = array(A, dtype=float)
    for i in range(n):
        chol[i, i] = chol[i, i]-sum(power(abs(chol[i, 0:i]), 2))
        if chol[i, i] >= 1e-15:
            chol[i, i] = sqrt(chol[i, i])
        else:
            return False, "facto_cholesky: no se factoriza la matriz"
        chol[i, i+1:] = (chol[i, i+1:] -
                         chol[i, 0:i]@conjugada(chol[i+1:, 0:i]))/chol[i, i]
        chol[i+1:, i] = conjugada(chol[i, i+1:])
    return True, chol


def metodo_cholesky(A, B):
    exito, chol = facto_cholesky(A)
    if exito:
        exito2, Y = descenso(chol, B)
        exito3, X = remonte(chol, Y)
        if exito2 and exito3:
            return True, X
        else:
            return False, "metodo_cholesky: error en la resolución."
    else:
        return False, "metodo_cholesky: error en la factorización."

# Matriz A,miembro B, vector X0 con el que iniciar las iteraciones
# y el numero maximo de iteraciones que dar


# A = M - N = D - (E + F) = D - E -F => J = (D^-1)(E+F)
# Sacar la J:  A = M - N = D - (E + F) = D - E -F => J = (D^-1)(E+F)
# D = A - tril(A, -1) - triu(A, 1)
# E = tril(-A, -1)
# F = triu(-A, 1)
# Dinv = inv(D)
# J = Dinv@(E+F)
def jacobi(A, B, XOLD, itermax, tol):
    (m, n) = shape(A)
    (p, q) = shape(B)
    (r, s) = shape(XOLD)
    if m != n or n != p or q != 1 or n != r or s != 1 or min(abs(diag(A))) < 1e-15:
        return False, 'ERROR jacobi: no se resuelve el sistema.'
    k = 0
    error = 1.
    while k < itermax and error >= tol:
        k = k+1
        XNEW = zeros((n, 1))
        for i in range(n):
            XNEW[i, 0] = B[i, 0]
            if i != 0:
                XNEW[i, 0] = XNEW[i, 0] - A[i, :i]@XOLD[:i, 0]
            if i != n-1:
                XNEW[i, 0] = XNEW[i, 0] - A[i, i+1:]@XOLD[i+1:, 0]
            XNEW[i, 0] = XNEW[i, 0]/A[i, i]
        error = norm(XNEW - XOLD, inf)
        XOLD = array(XNEW, copy=True)
    print('Iteración: k = ', k)
    print('Error absoluto: error = ', error)
    if k == itermax and error >= tol:
        return False, 'ERROR jacobi: no se resuelve el sistema.'
    else:
        print('Convergencia numérica alcanzada: jacobi.')
        return True, XNEW


def gaussseidel(A, B, XOLD, itermax, tol):
    (m, n) = shape(A)
    (p, q) = shape(B)
    (r, s) = shape(XOLD)
    if m != n or n != p or q != 1 or n != r or s != 1 or min(abs(diag(A))) < 1e-15:
        return False, 'ERROR jacobi: no se resuelve el sistema.'
    k = 0
    error = 1.
    while k < itermax and error >= tol:
        k = k+1
        XNEW = zeros((n, 1))
        for i in range(n):
            XNEW[i, 0] = B[i, 0]
            if i != 0:
                XNEW[i, 0] = XNEW[i, 0] - A[i, :i]@XNEW[:i, 0]
            if i != n-1:
                XNEW[i, 0] = XNEW[i, 0] - A[i, i+1:]@XOLD[i+1:, 0]
            XNEW[i, 0] = XNEW[i, 0]/A[i, i]
        error = norm(XNEW - XOLD, inf)
        XOLD = array(XNEW, copy=True)
    print('Iteración: k = ', k)
    print('Error absoluto: error = ', error)
    if k == itermax and error >= tol:
        return False, 'ERROR gaussseidel: no se resuelve el sistema.'
    else:
        print('Convergencia numérica alcanzada: gausseidel.')
        return True, XNEW

# omega es el omga de relajacion. Converge si ...


def relajacion(A, B, XOLD, omega, itermax, tol):
    (m, n) = shape(A)
    (p, q) = shape(B)
    (r, s) = shape(XOLD)
    if m != n or n != p or q != 1 or n != r or s != 1 or min(abs(diag(A))) < 1e-10:
        return False, 'ERROR relajación: no se resuelve el sistema.'
    k = 0
    error = 1.
    while k < itermax and error >= tol:
        k = k+1
        XNEW = zeros((n, 1))
        for i in range(n):
            XNEW[i, 0] = B[i, 0]
            if i != 0:
                XNEW[i, 0] = XNEW[i, 0] - A[i, :i]@XNEW[:i, 0]
            XNEW[i, 0] = XNEW[i, 0] + (1 - omega)/omega*A[i, i]*XOLD[i, 0]
            if i != n-1:
                XNEW[i, 0] = XNEW[i, 0] - A[i, i+1:]@XOLD[i+1:, 0]
            XNEW[i, 0] = omega*XNEW[i, 0]/A[i, i]
        error = norm(XNEW - XOLD, inf)
        XOLD = array(XNEW, copy=True)
    print('Iteración: k = ', k)
    print('Error: error = ', error)
    if k == itermax and error >= tol:
        return False, 'ERROR relajación: no se resuelve el sistema.'
    else:
        print('Convergencia numérica alcanzada: relajación.')
        return True, XNEW


# exito, normas, lambdas, X  = potencia(A, X0, inf, 200, 1e-8)
def potencia(A, X, norma, itermax, tol):
    (m, n) = shape(A)
    (r, s) = shape(X)
    if m != n or n != r or s != 1:
        return False, 'ERROR potencia: no se ejecuta el programa.', 0, 0
    k = 0
    error = 1.
    normaold = 0.
    lambdas = zeros(n, dtype=complex)
    while k < itermax and error >= tol:
        k = k+1
        Y = A@X
        normanew = norm(Y, ord=norma)
        error = abs(normanew - normaold)
        for i in range(n):
            if abs(X[i, 0]) >= 1.e-15:
                lambdas[i] = Y[i, 0]/X[i, 0]
            else:
                lambdas[i] = 0.
        X = Y/normanew
        #   
        # print('Lambdas: lambdas_k = \n', lambdas)
        # print('Vectores: X_k = \n', X)
        normaold = normanew
    if k == itermax and error >= tol:
        return False, 'ERROR potencia: no se alcanza convergencia.', 0, 0
    else:
        print('\n Potencia: convergencia numérica alcanzada.')
        return True, normanew, lambdas, X


def potenciainv(A, X, norma, itermax, tol):
    (m, n) = shape(A)
    (r, s) = shape(X)
    if m != n or n != r or s != 1:
        return False, 'ERROR potenciainv: no se ejecuta el programa.', 0, 0
    exito, LU = facto_lu(A)
    if not exito:
        return False, 'ERROR potenciainv: sin factorización LU.', 0, 0
    k = 0
    error = 1.
    normaold = 0.
    lambdas = zeros(n, dtype=complex)
    while k < itermax and error >= tol:
        k = k+1
        exito, Y = descenso1(LU, X)
        exito, Y = remonte(LU, Y)
        if not exito:
            return False, 'ERROR potenciainv: sin factorización LU.', 0, 0
        normanew = norm(Y, ord=norma)
        error = abs(normanew - normaold)
        for i in range(n):
            if abs(X[i, 0]) >= 1e-15:
                lambdas[i] = Y[i, 0]/X[i, 0]
            else:
                lambdas[i] = 0.
        X = Y/normanew
        # print('Iteración: k = ', k, 'Norma: ||A-1*X_k|| = ', normanew)
        # print('Lambdas: lambdas_k = ', lambdas)
        # print('Vectores: X_k = ', X)
        normaold = normanew
    if k == itermax and error >= tol:
        return False, 'ERROR potenciainv: no se alcanza convergencia.', 0, 0
    else:
        print('\n Potenciainv: convergencia numérica alcanzada.')
        return True, normanew, lambdas, X


def potenciades(A, X, des, norma, itermax, tol):
    (m, n) = shape(A)
    (r, s) = shape(X)
    if m != n or n != r or s != 1:
        return False, 'ERROR potenciades: no se ejecuta el programa.'
    B = A-des*eye(n)
    exito, normanew, lambdas, X = potencia(B, X, norma, itermax, tol)
    return exito, normanew, lambdas, X


def potenciadesinv(A, X, des, norma, itermax, tol):
    (m, n) = shape(A)
    (r, s) = shape(X)
    if m != n or n != r or s != 1:
        return False, 'ERROR potenciadesinv: no se ejecuta el programa.'
    B = A-des*eye(n)
    exito, normanew, lambdas, X = potenciainv(B, X, norma, itermax, tol)
    return exito, normanew, lambdas, X


def Householder(X):
    den = conjugada(X)@X
    N = X@conjugada(X)
    fr = 2/den
    n = size(X)
    H = eye(n, n) - fr * N
    return H


def matricesHouseholder(X):
    num = e**(angle(X[0])*1j)
    n = size(X)
    A = X + norma_vec(X, 2)*num*eye(n, 1)
    B = X - norma_vec(X, 2)*num*eye(n, 1)
    return Householder(A), Householder(B)


def matrizTriangularGauss(A, B):
    m, n = shape(A)
    p, q = shape(B)
    if m != n or n != p or q < 1:
        return False, "Error gauss_pp: error en las dimensiones."
    if A.dtype == complex or B.dtype == complex:
        gaussA = array(A, dtype=complex)
        gaussB = array(B, dtype=complex)
    else:
        gaussA = array(A, dtype=float)
        gaussB = array(B, dtype=float)
    for k in range(n-1):
        pos = argmax(abs(gaussA[k:, k]))
        ik = pos+k
        if ik != k:
            gaussA[[ik, k], :] = gaussA[[k, ik], :]
            gaussB[[ik, k], :] = gaussB[[k, ik], :]
        if abs(gaussA[k, k]) >= 1e-200:
            for i in range(k+1, n):
                gaussA[i, k] = gaussA[i, k]/gaussA[k, k]
                gaussA[i, k+1:] -= gaussA[i, k]*gaussA[k, k+1:]
                gaussB[i, :] -= gaussA[i, k]*gaussB[k, :]
    return triu(gaussA)


def cond(A, p):
    n, m = shape(A)
    if m != n:
        return "Error cond"
    AA = array(A, dtype=complex)
    if p == 1:
        return norma_mat(AA, 1)*norma_mat(inv(AA), 1)
    elif p == 2:
        return norma_mat(AA, 2)*norma_mat(inv(AA), 2)
    elif p == inf:
        return norma_mat(AA, inf)*norma_mat(inv(AA), inf)



# funciones a conocer
# ndim(A) dimensiones de una matriz
# shape(A) tamaño de una matriz (_,_)
# size(A) numero de elementos de una matriz
# zeros(n) matriz de ceros    (zeros([3,3]) matriz 3x3 de ceros
# ones(n) matriz de unos
# eye(n) matriz con diagonal 1, n es la dimension de la matriz
# transpose(A) matriz transpuesta
# tril(A) triangular inferior
# tril(A) triangular superior
# diag(A) vector de la diagonal
# diagflat(A) matriz diagonal a partir de un vector
# traza(A) traza o trace
# svd(A), matriz U, valores singulares y matriz V
#   
