import numpy as np  
from numpy import linalg  

def sanssenfeld(A):
    n = np.shape(A)[0]
    B = np.zeros(n)

    # Inicializa o primeiro valor de B
    B[0] = np.sum(np.abs(A[0])) - np.abs(A[0][0])
    
    for i in range(1,n):
        a = 0
        for j in range(0,n):
            if i != j and a < j:
                B[i] += abs(A[i][j]) * B[a]
            elif i != j:
                B[i] += abs(A[i][j])
        

    # Pega o maior valor de x
    max_x = max(B)
    print(max_x)
    return max_x < 1


def converge(A):
    n = np.shape(A)[0] 

    for i in range(0, n):
        sum = 0
        for j in range(0,n):
            if i != j:
                sum += A[i][j]
        if abs(A[i][i]) < sum:
            print(sum)
            print(A[i][i])
            return False
        
    return True

def jacobi(A,b,x0,tol,N):  

    c = converge(A)
    if not c:
        c = sanssenfeld(A)
        if not c:
            return -1

    n = np.shape(A)[0]  
    x = np.zeros(n)  
    it = 0 

    #iteracoes  
    while (it < N):  
        it += 1

        #iteracao de Jacobi  
        for i in range(0, n):  
            x[i] = b[i]  
            for j in range(0, n):  
                if i != j:
                    x[i] -= A[i,j] * x0[j]  
            x[i] /= A[i,i] 

        #tolerancia  
        if (np.linalg.norm(x-x0,np.inf) < tol):  
            return x  
        
        #prepara nova iteracao  
        x0 = np.copy(x)  

    return x
