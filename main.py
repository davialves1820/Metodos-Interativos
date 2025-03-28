import numpy as np
from jacobi import jacobi

def main():
    # Define o sistema de equações: Ax = b
    A = np.array([[-3, 1, 1],[2, 5, 1],[2, 3, 7]], dtype=float)
    b = np.array([2, 5, -17], dtype=float)
    x0 = np.array([1,1,1], dtype=float)  # Chute inicial
    tol = 0.006  # Tolerância
    N = 10  # Número máximo de iterações

    try:
        solucao = jacobi(A, b, x0, tol, N)
        print("Solução encontrada:", solucao)
    except ValueError as e:
        print("Erro:", e)

# Executa o programa
if __name__ == "__main__":
    main()
