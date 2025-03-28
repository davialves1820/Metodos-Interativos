import numpy as np  

def gauss_seidel(A,b,x0,tol):  
    #preliminares  
    A = A.astype("double")  
    b = b.astype("double")  
    x0 = x0.astype("double")  

    n=np.shape(A)[0]  
    x = np.copy(x0)  
    it = 0  
    #iteracoes  
    while True:  

        #iteracao de Jacobi  
        for i in np.arange(n):  
            x[i] = b[i]  
            for j in np.concatenate((np.arange(0,i),np.arange(i+1,n))):  
                x[i] -= A[i,j]*x[j]  
            x[i] /= A[i,i]  
            print(x[i],A[i,i])  
        #tolerancia  
        if (np.linalg.norm(x-x0,np.inf) < tol):  
            return x  
        #prepara nova iteracao  
        x0 = np.copy(x)  